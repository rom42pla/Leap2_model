from abc import abstractmethod
import itertools
import math
from pprint import pprint
from types import FunctionType
import torch
import torch.nn.functional as F
import lightning as pl
import torchmetrics
from typing import Any, Callable, Tuple
from torch import nn
from tqdm import tqdm
import timm
from transformers import CLIPVisionModel, AutoModel


class Model(pl.LightningModule):
    _possible_data_available = {
        "only_both_images",
        "only_one_image",
        "only_both_landmarks",
        "only_one_kind_of_landmarks",
        "both_images_and_landmarks",
    }
    _possible_landmarks_backbones = {None, "unprocessed", "linear", "mlp"}
    _possible_image_backbones = {None, "resnet", "clip", "dinov2"}
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
        data_available: str = "both_images_and_landmarks",
        image_backbone_name: str = "clip",
        landmarks_backbone_name: str = "mlp",
        merging_method: str = "concatenate",
        h_dim: int = 512,
        lr: float = 5e-5,
        **kwargs,
    ):
        super(Model, self).__init__()
        assert (
            data_available in self._possible_data_available
        ), f"got {data_available}, expected one of {self._possible_data_available}"
        self.data_available = data_available
        if "image" in self.data_available:
            assert (
                image_backbone_name is not None
            ), f"got mode '{self.data_available}' but got no image backbone name"
        if "landmarks" in self.data_available:
            assert (
                landmarks_backbone_name is not None
            ), f"got mode '{self.data_available}' but got no landmarks backbone name"

        self.num_classes = num_labels
        self.num_landmarks = num_landmarks
        self.img_channels = img_channels
        self.img_size = img_size

        # model params
        assert isinstance(h_dim, int) and h_dim > 0, h_dim
        self.h_dim = h_dim

        # image backbone
        self.image_backbone_name = image_backbone_name
        if self.image_backbone_name:
            self.adapter = nn.Conv2d(
                in_channels=self.img_channels
                * (2 if "both" in self.data_available else 1),
                out_channels=3,
                kernel_size=11,
                stride=1,
                padding="same",
            )
        self.image_features_size, self.image_backbone, self.image_embedder = (
            self._parse_image_backbone(name=self.image_backbone_name)
        )

        # landmarks backbone
        self.landmarks_backbone_name = landmarks_backbone_name
        (
            self.landmarks_features_size,
            self.landmarks_backbone,
            self.landmarks_embedder,
        ) = self._parse_landmarks_backbone(name=self.landmarks_backbone_name)

        # classifier
        self.merging_method_name = merging_method
        (
            self.image_features_embedder,
            self.landmarks_features_embedder,
            self.cls_head,
            self.classify,
        ) = self._parse_merging_method(name=self.merging_method_name)

        # optimizer params
        assert lr > 0
        self.lr = lr

        self.epoch_metrics = {}

        self.save_hyperparameters(ignore=["epoch_metrics"])

    # returns image features size, the image backbone and the function to call the backbone
    def _parse_image_backbone(self, name) -> Tuple[int, nn.Module, Callable]:
        assert (
            name in self._possible_image_backbones
        ), f"got {name}, expected one of {self._possible_image_backbones}"
        if name is None:
            return 0, nn.Identity(), nn.Identity()
        elif name == "resnet":
            image_features_size = 512
            image_backbone = timm.create_model("resnet18.a1_in1k", pretrained=True)
            image_backbone.fc = nn.Identity()
            image_embedder = lambda imgs: image_backbone(self.adapter(imgs))
        elif name == "clip":
            image_features_size = 768
            image_backbone = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            image_embedder = lambda imgs: image_backbone(
                pixel_values=self.adapter(imgs)
            ).pooler_output
        elif name == "dinov2":
            image_features_size = 768
            image_backbone = AutoModel.from_pretrained("facebook/dinov2-base")
            image_embedder = lambda imgs: image_backbone(
                pixel_values=self.adapter(imgs)
            ).pooler_output
        else:
            raise NotImplementedError(
                f"got image backbone '{name}', expected one of {self._possible_image_backbones}"
            )
        # freezes the backbone
        for param in image_backbone.parameters():
            param.requires_grad = False
        return image_features_size, image_backbone, image_embedder

    def _parse_landmarks_backbone(self, name) -> Tuple[int, nn.Module, Callable]:
        landmarks_features_size = 768
        if name is None:
            return 0, nn.Identity(), nn.Identity()
        if name == "unprocessed":
            landmarks_features_size = self.num_landmarks
            landmarks_backbone = nn.Identity()
        elif name == "linear":
            landmarks_backbone = nn.Linear(self.num_landmarks, landmarks_features_size)
        elif name == "mlp":
            landmarks_backbone = nn.Sequential(
                nn.Linear(self.num_landmarks, 512 * 4),
                nn.LeakyReLU(),
                nn.Linear(512 * 4, landmarks_features_size),
            )
        else:
            raise NotImplementedError(
                f"got landmarks backbone '{name}', expected one of {self._possible_landmarks_backbones}"
            )
        landamarks_embedder = lambda landmarks: landmarks_backbone(landmarks)
        return landmarks_features_size, landmarks_backbone, landamarks_embedder

    def _parse_merging_method(
        self, name
    ) -> Tuple[nn.Module, nn.Module, nn.Module, Callable]:
        image_features_embedder, landmarks_features_embedder = (
            nn.Identity(),
            nn.Identity(),
        )
        if name == "concatenate":
            cls_head = nn.Linear(
                self.image_features_size + self.landmarks_features_size,
                self.num_classes,
            )

            def classify(image_features, landmarks_features):
                assert (image_features is not None) or (landmarks_features is not None)
                tensors = []
                if image_features is not None:
                    tensors.append(image_features)
                if landmarks_features is not None:
                    tensors.append(landmarks_features)
                return cls_head(torch.concatenate(tensors, dim=1))

        elif name == "sum":
            h_dim = max(self.image_features_size, self.landmarks_features_size)
            cls_head = nn.Linear(h_dim, self.num_classes)
            if self.image_features_size != h_dim:
                image_features_embedder = nn.Linear(self.image_features_size, h_dim)
            elif self.landmarks_features_size != h_dim:
                landmarks_features_embedder = nn.Linear(
                    self.landmarks_features_size, h_dim
                )

            def classify(image_features, landmarks_features):
                assert (image_features is not None) or (landmarks_features is not None)
                tensors = []
                if image_features is not None:
                    tensors.append(image_features_embedder(image_features))
                if landmarks_features is not None:
                    tensors.append(landmarks_features_embedder(landmarks_features))
                return cls_head(sum(tensors))

        else:
            raise NotImplementedError(
                f"merging method {self._merging_method} not implemented"
            )
        return image_features_embedder, landmarks_features_embedder, cls_head, classify

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        keys_to_pop = []
        for module_name in checkpoint["state_dict"].keys():
            if module_name.startswith("_image_backbone"):
                keys_to_pop.append(module_name)
        for module_name in keys_to_pop:
            del checkpoint["state_dict"][module_name]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.image_features_size, self.image_backbone, self.image_embedder = (
            self._parse_image_backbone(name=self.image_backbone_name)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    @abstractmethod
    def forward(
        self,
        landmarks_horizontal=None,
        landmarks_vertical=None,
        image_horizontal=None,
        image_vertical=None,
        **kwargs,
    ):
        outs = {
            "imgs_embs": None,
            "landmarks_embs": None,
        }
        if self.image_backbone_name is not None:
            t = []
            if image_horizontal is not None:
                t.append(image_horizontal)
            if image_vertical is not None:
                t.append(image_vertical)
            # eventually uses just one image
            if self.data_available == "only_one_image" and len(t) > 1:
                 t = [t[0]]
            imgs = torch.cat(t, dim=1).float().to(self.device)
            outs["imgs_embs"] = self.image_embedder(imgs)

        if self.landmarks_backbone_name is not None:
            t = []
            if landmarks_horizontal is not None:
                t.append(landmarks_horizontal)
            if landmarks_vertical is not None:
                t.append(landmarks_vertical)
            # eventually uses just one kind of landmarks
            if self.data_available == "only_one_kind_of_landmarks" and len(t) > 1:
                t = [t[0]]
            landmarks = torch.cat(t, dim=1).float().to(self.device)
            outs["landmarks_embs"] = self.landmarks_embedder(landmarks)
        outs["cls_logits"] = self.classify(
            image_features=outs["imgs_embs"], landmarks_features=outs["landmarks_embs"]
        )
        return outs

    def step(self, batch, phase):
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
    # define params for the tests
    params = {
        "num_labels": 17,
        "batch_size": 16,
        "num_landmarks": 242,
        "img_channels": 1,
        "img_size": 224,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # define a dummy batch
    dummy_batch = {
        "landmarks_horizontal": torch.rand(
            [params["batch_size"], params["num_landmarks"]], device=params["device"]
        ),
        "landmarks_vertical": torch.rand(
            [params["batch_size"], params["num_landmarks"]], device=params["device"]
        ),
        "image_horizontal": torch.rand(
            [
                params["batch_size"],
                params["img_channels"],
                params["img_size"],
                params["img_size"],
            ],
            device=params["device"],
        ),
        "image_vertical": torch.rand(
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
    for (
        data_available,
        landmarks_backbone_name,
        image_backbone_name,
        merging_method,
    ) in tqdm(
        list(
            itertools.product(
                Model._possible_data_available,
                Model._possible_landmarks_backbones,
                Model._possible_image_backbones,
                Model._possible_merging_methods,
            )
        ),
        desc="trying all backbones combinations",
    ):
        if (
            not (landmarks_backbone_name or image_backbone_name)
            or ("landmarks" in data_available and not landmarks_backbone_name)
            or ("image" in data_available and not image_backbone_name)
        ):
            continue
        # try:
        print(
            data_available, image_backbone_name, landmarks_backbone_name, merging_method
        )
        model = Model(
            data_available=data_available,
            image_backbone_name=image_backbone_name,
            landmarks_backbone_name=landmarks_backbone_name,
            merging_method=merging_method,
            **params,
        ).to(params["device"])
        model.step(batch=dummy_batch, phase="train")
        del model
        # except Exception as e:
        #     print(
        #         f"combination {landmarks_backbone_name, image_backbone_name, merging_method} failed with error {e}"
        #     )
