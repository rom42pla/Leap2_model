from abc import abstractmethod
import math
from pprint import pprint
import torch
import torch.nn.functional as F
import lightning as pl
import torchmetrics
from typing import Optional, Union, List
from torch import nn
from transformers import CLIPVisionModel


class Model(pl.LightningModule):
    def __init__(self,
                 num_labels: int,
                 num_landmarks: int,
                 img_channels: int,
                 img_size: int,
                 h_dim: int = 512,
                 lr: float = 5e-5):
        super(Model, self).__init__()

        self.num_classes = num_labels
        self.num_landmarks = num_landmarks
        self.img_channels = img_channels
        self.img_size = img_size

        # model params
        assert isinstance(h_dim, int) and h_dim > 0, h_dim
        self.h_dim = h_dim

        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip.parameters(): 
            param.requires_grad = False
        self.clip.vision_model.embeddings.patch_embedding = nn.Conv2d(self.img_channels * 2, 768, kernel_size=32, stride=32, bias=False) # type: ignore
        for param in self.clip.vision_model.embeddings.patch_embedding.parameters(): # type: ignore
            param.requires_grad = True 

        self.landmarks_embedder = nn.Sequential(
            nn.Linear(self.num_landmarks, 768*4),
            nn.LeakyReLU(),
            nn.Linear(768*4, 768),
            )
        self.cls_head = nn.Linear(768*2, self.num_classes)

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
    def forward(self, landmarks_horizontal, landmarks_vertical, image_horizontal_left, image_vertical_left, **kwargs):
        outs = {}
        imgs = torch.cat([image_horizontal_left, image_vertical_left], dim=1).float().to(self.device)
        landmarks = torch.cat([landmarks_horizontal, landmarks_vertical], dim=1).float().to(self.device)

        outs["imgs_embs"] = self.clip(pixel_values=imgs).pooler_output
        outs["landmarks_embs"] = self.landmarks_embedder(landmarks)

        outs["cls_logits"] = self.cls_head(
            torch.cat([outs["imgs_embs"], outs["landmarks_embs"]], dim=1)
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
            "cls_prec": torchmetrics.functional.precision(preds=logits_stacked, target=labels_stacked, task="multiclass", num_classes=self.num_classes, average="micro"),
            "cls_rec": torchmetrics.functional.recall(preds=logits_stacked, target=labels_stacked, task="multiclass", num_classes=self.num_classes, average="micro"),
            "cls_acc": torchmetrics.functional.accuracy(preds=logits_stacked, target=labels_stacked, task="multiclass", num_classes=self.num_classes, average="micro"),
            "cls_f1": torchmetrics.functional.f1_score(preds=logits_stacked, target=labels_stacked, task="multiclass", num_classes=self.num_classes, average="micro"),
        }
        for metric_name, metric_value in metrics.items():
            self.log(name=f"{metric_name}_{phase}", value=metric_value,
                     prog_bar=True if "f1" in metric_name else False)
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