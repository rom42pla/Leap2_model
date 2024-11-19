from abc import abstractmethod
import math
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
        self.save_hyperparameters()
        
        self.num_classes = num_labels
        self.num_landmarks = num_landmarks
        self.img_channels = img_channels
        self.img_size = img_size

        # model params
        assert isinstance(h_dim, int) and h_dim > 0, h_dim
        self.h_dim = h_dim
        
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.vision_model.embeddings.patch_embedding = nn.Conv2d(self.img_channels * 2, 768, kernel_size=32, stride=32, bias=False)
        
        self.landmarks_embedder = nn.Sequential(
            nn.Linear(self.num_landmarks, 768*4),
            nn.LeakyReLU(),
            nn.Linear(768*4, 768),
            )
        self.cls_head = nn.Linear(768*2, self.num_classes)

        # optimizer params
        assert lr > 0
        self.lr = lr

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    @abstractmethod
    def forward(self, landmarks_horizontal, landmarks_vertical, horizontal_image, vertical_image, **kwargs):
        outs = {}
        imgs = torch.cat([horizontal_image, vertical_image], dim=1).float()
        landmarks = torch.cat([landmarks_horizontal, landmarks_vertical], dim=1).float()
        
        outs["imgs_embs"] = self.clip(pixel_values=imgs).pooler_output
        outs["landmarks_embs"] = self.landmarks_embedder(landmarks)
        
        outs["cls_logits"] = self.cls_head(
            torch.cat([outs["imgs_embs"], outs["landmarks_embs"]], dim=1)
        )
        return outs

    def step(self, batch, phase):
        outs = {
            "metrics": {},
            "cls_labels": batch["label"],
            **self(**batch),
        }

        outs["metrics"].update({
            "cls_loss": F.cross_entropy(input=outs["cls_logits"], target=outs["cls_labels"]),
            "cls_acc": torchmetrics.functional.accuracy(preds=outs["cls_logits"], target=outs["cls_labels"], task="multiclass", num_classes=self.num_classes, average="micro"),
            "cls_f1": torchmetrics.functional.f1_score(preds=outs["cls_logits"], target=outs["cls_labels"], task="multiclass", num_classes=self.num_classes, average="micro"),
        })


        # computes final loss
        outs["loss"] = sum(
            [v for k, v in outs["metrics"].items() if k.endswith("loss") and v.numel() == 1])

        # logs metrics
        for metric_name, metric_value in outs["metrics"].items():
            self.log(name=f"{metric_name}/{phase}", value=metric_value,
                     prog_bar=any([metric_name.endswith(s)
                                  for s in {"f1"}]),
                     on_step=False, on_epoch=True, batch_size=batch["label"].shape[0])
        return outs

    def training_step(self, batch, batch_idx):
        outs = self.step(batch, phase="train")
        return outs

    def validation_step(self, batch, batch_idx):
        outs = self.step(batch, phase="val")
        return outs

    def test_step(self, batch, batch_idx):
        outs = self.step(batch, phase="test")
        return outs
