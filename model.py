from abc import abstractmethod
import itertools
import math
from pprint import pprint
import torch
import torch.nn.functional as F
import lightning as pl
import torchmetrics
from typing import Optional, Union, List
from torch import nn
from tqdm import tqdm
import timm
from transformers import CLIPVisionModel, AutoModel


class Model(pl.LightningModule):
    _possible_landmarks_backbones = {"none", "linear", "mlp"}
    _possible_image_backbones = {"resnet", "clip", "dinov2"}
    _possible_merging_methods = {
        "concatenate",
        "sum",
    }

    def __init__(
        self,
        num_labels: int,
        num_landmarks: int,
        img_channels: int,
        img_size: int,
        image_backbone_name: str = "clip",
        landmarks_backbone_name: str = "mlp",
        merging_method: str = "concatenate",
        h_dim: int = 512,
        lr: float = 5e-5,
        **kwargs,
    ):
        super(Model, self).__init__()

        self.num_classes = num_labels
        self.num_landmarks = num_landmarks
        self.img_channels = img_channels
        self.img_size = img_size

        # model params
        assert isinstance(h_dim, int) and h_dim > 0, h_dim
        self.h_dim = h_dim

        """
        IMAGES 
        BRANCH
        """
        assert (
            image_backbone_name in self._possible_image_backbones
        ), f"got {image_backbone_name}, expected one of {self._possible_image_backbones}"
        self._image_backbone_name = image_backbone_name
        self._adapter = nn.Conv2d(
            in_channels=self.img_channels * 2,
            out_channels=3,
            kernel_size=11,
            stride=1,
            padding="same",
        )
        if self._image_backbone_name == "resnet":
            self.image_features_size = 512
            self._image_backbone = timm.create_model(
                "resnet18.a1_in1k", pretrained=True
            )
            self._image_backbone.fc = nn.Identity()
            self.image_embedder = lambda imgs: self._image_backbone(self._adapter(imgs))
        elif self._image_backbone_name == "clip":
            self.image_features_size = 768
            self._image_backbone = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            # self.clip.vision_model.embeddings.patch_embedding = nn.Conv2d(self.img_channels * 2, 768, kernel_size=32, stride=32, bias=False) # type: ignore
            # for param in self.clip.vision_model.embeddings.patch_embedding.parameters(): # type: ignore
            #     param.requires_grad = True
            self.image_embedder = lambda imgs: self._image_backbone(
                pixel_values=self._adapter(imgs)
            ).pooler_output
        elif self._image_backbone_name == "dinov2":
            self.image_features_size = 768
            self._image_backbone = AutoModel.from_pretrained("facebook/dinov2-base")
            self.image_embedder = lambda imgs: self._image_backbone(
                pixel_values=self._adapter(imgs)
            ).pooler_output
        else:
            raise NotImplementedError(
                f"image backbone {self._image_backbone_name} not implemented"
            )
        # freezes the backbone
        for param in self._image_backbone.parameters():
            param.requires_grad = False

        """
        LANDMARKS 
        BRANCH
        """
        assert (
            landmarks_backbone_name in self._possible_landmarks_backbones
        ), f"got {landmarks_backbone_name}, expected one of {self._possible_landmarks_backbones}"
        self._landmarks_backbone_name = landmarks_backbone_name
        self.landmarks_features_size = 768
        if self._landmarks_backbone_name == "none":
            self.landmarks_features_size = self.num_landmarks
            self._landmarks_backbone = nn.Identity()
        elif self._landmarks_backbone_name == "linear":
            self._landmarks_backbone = nn.Linear(
                self.num_landmarks, self.landmarks_features_size
            )
        elif self._landmarks_backbone_name == "mlp":
            self._landmarks_backbone = nn.Sequential(
                nn.Linear(self.num_landmarks, 512 * 4),
                nn.LeakyReLU(),
                nn.Linear(512 * 4, self.landmarks_features_size),
            )
        else:
            raise NotImplementedError(
                f"landmarks backbone {self._landmarks_backbone_name} not implemented"
            )
        self.landmarks_embedder = lambda landmarks: self._landmarks_backbone(landmarks)

        """
        MERGING 
        BRANCH
        """
        assert (
            merging_method in self._possible_merging_methods
        ), f"got {merging_method}, expected one of {self._possible_merging_methods}"
        self._merging_method = merging_method
        if self._merging_method == "concatenate":
            self._cls_head = nn.Linear(
                self.image_features_size + self.landmarks_features_size,
                self.num_classes,
            )
            self.classify = lambda image_features, landmarks_features: self._cls_head(
                torch.concatenate([image_features, landmarks_features], dim=1)
            )
        elif self._merging_method == "sum":
            h_dim = max(self.image_features_size, self.landmarks_features_size)
            self._cls_head = nn.Linear(h_dim, self.num_classes)
            self.image_features_embedder, self.landmarks_features_embedder = (
                nn.Identity(),
                nn.Identity(),
            )
            if self.image_features_size != h_dim:
                self.image_features_embedder = nn.Linear(
                    self.image_features_size, h_dim
                )
            elif self.landmarks_features_size != h_dim:
                self.landmarks_features_embedder = nn.Linear(
                    self.landmarks_features_size, h_dim
                )
            self.classify = lambda image_features, landmarks_features: self._cls_head(
                self.image_features_embedder(image_features)
                + self.landmarks_features_embedder(landmarks_features)
            )
        else:
            raise NotImplementedError(
                f"merging method {self._merging_method} not implemented"
            )

        # optimizer params
        assert lr > 0
        self.lr = lr

        self.save_hyperparameters(ignore="epoch_metrics")

        self.epoch_metrics = {}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    @abstractmethod
    def forward(
        self,
        landmarks_horizontal,
        landmarks_vertical,
        image_horizontal_left,
        image_vertical_left,
        **kwargs,
    ):
        outs = {}
        imgs = (
            torch.cat([image_horizontal_left, image_vertical_left], dim=1)
            .float()
            .to(self.device)
        )
        landmarks = (
            torch.cat([landmarks_horizontal, landmarks_vertical], dim=1)
            .float()
            .to(self.device)
        )

        outs["imgs_embs"] = self.image_embedder(imgs)
        outs["landmarks_embs"] = self.landmarks_embedder(landmarks)
        outs["cls_logits"] = self.classify(
            image_features=outs["imgs_embs"], landmarks_features=outs["landmarks_embs"]
        )
        return outs

    def step(self, batch, phase):
        # for k, v in batch.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.shape)
        outs = self(**batch)
        outs["loss"] = F.cross_entropy(input=outs["cls_logits"], target=batch["label"])
        if not phase in self.epoch_metrics:
            self.epoch_metrics[phase] = {}
        for key in ["cls_labels", "cls_logits", "loss"]:
            if not key in self.epoch_metrics[phase]:
                self.epoch_metrics[phase][key] = []
        self.epoch_metrics[phase]["cls_labels"].append(batch["label"])
        self.epoch_metrics[phase]["cls_logits"].append(outs["cls_logits"])
        self.epoch_metrics[phase]["loss"].append(outs["loss"])

        # outs = {
        #     # "metrics": {},
        #     "cls_labels": batch["label"],
        #     **self(**batch),
        # }

        # outs["metrics"].update({
        #     "cls_loss": F.cross_entropy(input=outs["cls_logits"], target=outs["cls_labels"]),
        #     "cls_prec": torchmetrics.functional.precision(preds=outs["cls_logits"], target=outs["cls_labels"], task="multiclass", num_classes=self.num_classes, average="micro"),
        #     "cls_rec": torchmetrics.functional.recall(preds=outs["cls_logits"], target=outs["cls_labels"], task="multiclass", num_classes=self.num_classes, average="micro"),
        #     "cls_acc": torchmetrics.functional.accuracy(preds=outs["cls_logits"], target=outs["cls_labels"], task="multiclass", num_classes=self.num_classes, average="micro"),
        #     "cls_f1": torchmetrics.functional.f1_score(preds=outs["cls_logits"], target=outs["cls_labels"], task="multiclass", num_classes=self.num_classes, average="micro"),
        # })

        # computes final loss
        # outs["loss"] = sum(
        #     [v for k, v in outs["metrics"].items() if k.endswith("loss") and v.numel() == 1])

        # logs metrics
        # for metric_name, metric_value in outs["metrics"].items():
        #     self.log(name=f"{metric_name}_{phase}", value=metric_value,
        #              prog_bar=any([metric_name.endswith(s)
        #                           for s in {"f1"}]),
        #              on_step=True, on_epoch=True, batch_size=batch["label"].shape[0])
        # return outs
        return outs

    def on_epoch_end(self, phase):
        logits_stacked = torch.cat(self.epoch_metrics[phase]["cls_logits"], dim=0)
        labels_stacked = torch.cat(self.epoch_metrics[phase]["cls_labels"], dim=0)
        metrics = {
            "cls_loss": torch.stack(self.epoch_metrics[phase]["loss"], dim=0).mean(),
            "cls_prec": torchmetrics.functional.precision(
                preds=logits_stacked,
                target=labels_stacked,
                task="multiclass",
                num_classes=self.num_classes,
                average="micro",
            ),
            "cls_rec": torchmetrics.functional.recall(
                preds=logits_stacked,
                target=labels_stacked,
                task="multiclass",
                num_classes=self.num_classes,
                average="micro",
            ),
            "cls_acc": torchmetrics.functional.accuracy(
                preds=logits_stacked,
                target=labels_stacked,
                task="multiclass",
                num_classes=self.num_classes,
                average="micro",
            ),
            "cls_f1": torchmetrics.functional.f1_score(
                preds=logits_stacked,
                target=labels_stacked,
                task="multiclass",
                num_classes=self.num_classes,
                average="micro",
            ),
        }
        for metric_name, metric_value in metrics.items():
            self.log(
                name=f"{metric_name}_{phase}",
                value=metric_value,
                prog_bar=True if "f1" in metric_name else False,
            )
        del self.epoch_metrics[phase]

    def training_step(self, batch, batch_idx):
        outs = self.step(batch, phase="train")
        return outs

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            outs = self.step(batch, phase="val")
        return outs

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            outs = self.step(batch, phase="test")
        return outs

    # def on_test_epoch_start(self):
    #     self.epoch_metrics = {
    #         "cls_labels": [],
    #         "cls_logits": [],
    #         "loss": [],
    #     }

    def on_train_epoch_end(self):
        self.on_epoch_end(phase="train")

    def on_validation_epoch_end(self):
        self.on_epoch_end(phase="val")

    def on_test_epoch_end(self):
        self.on_epoch_end(phase="test")


if __name__ == "__main__":
    params = {
        "num_labels": 17,
        "batch_size": 16,
        "num_landmarks": 242,
        "img_channels": 1,
        "img_size": 224,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    dummy_batch = {
        "landmarks_horizontal": torch.rand(
            [params["batch_size"], params["num_landmarks"]], device=params["device"]
        ),
        "landmarks_vertical": torch.rand(
            [params["batch_size"], params["num_landmarks"]], device=params["device"]
        ),
        "image_horizontal_left": torch.rand(
            [
                params["batch_size"],
                params["img_channels"],
                params["img_size"],
                params["img_size"],
            ],
            device=params["device"],
        ),
        "image_vertical_left": torch.rand(
            [
                params["batch_size"],
                params["img_channels"],
                params["img_size"],
                params["img_size"],
            ],
            device=params["device"],
        ),
        "label": torch.zeros(
            [params["batch_size"], params["num_labels"]], device=params["device"]
        ),
    }
    params["num_landmarks"] *= 2
    for landmarks_backbone_name, image_backbone_name, merging_method in tqdm(
        list(
            itertools.product(
                Model._possible_landmarks_backbones,
                Model._possible_image_backbones,
                Model._possible_merging_methods,
            )
        ),
        desc="trying all backbones combinations",
    ):
        try:
            model = Model(
                image_backbone_name=image_backbone_name,
                landmarks_backbone_name=landmarks_backbone_name,
                merging_method=merging_method,
                **params,
            ).to(params["device"])
            model.step(batch=dummy_batch, phase="train")
        except Exception as e:
            print(
                f"combination {landmarks_backbone_name, image_backbone_name, merging_method} failed with error {e}"
            )
