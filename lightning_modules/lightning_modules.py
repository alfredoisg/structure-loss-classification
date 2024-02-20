import pytorch_lightning as pl

import torch
import torch.nn as nn
import torchmetrics

from models.models import LeNet5

import numpy as np


class LitModelBase(pl.LightningModule):
    def __init__(
        self, num_classes: int, learning_rate: float, pretrained: bool = False
    ) -> None:
        super().__init__()

        self.num_classes = num_classes


        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

        self.cm = torchmetrics.ConfusionMatrix(task="multiclass",
                                               num_classes=num_classes)

        self.stored_confusion_matrix = None

        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.pretrained = pretrained

    def _common_step(self, batch, batch_idx, stage=None):
        inputs, labels = batch

        predictions = self(inputs)

        loss = self.loss_fn(predictions, labels)

        accuracy = self.accuracy(predictions, labels)
        f1_score = self.f1_score(predictions, labels)

        self.cm(predictions, labels)

        if stage:
            self.log_dict(
                {
                    f"{stage}_loss": loss,
                    f"{stage}_accuracy": accuracy,
                    f"{stage}_f1_score": f1_score,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="val")


    def on_validation_epoch_end(self):
       
        cm = self.cm.compute()  
        self.stored_confusion_matrix = cm.cpu().numpy()
        print("Validation Confusion Matrix:", cm)
        
        self.cm.reset()



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LitLeNet5(LitModelBase):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float,
        size_layer_1: int = 84,
        size_layer_2: int = 10,
        pretrained: bool = False,
    ) -> None:
        super().__init__(
            num_classes,
            learning_rate,
            pretrained=pretrained,
        )

        self.size_layer_1 = size_layer_1
        self.size_layer_2 = size_layer_2

        self.model = LeNet5(
            num_classes=num_classes,
            size_layer_1=size_layer_1,
            size_layer_2=size_layer_2,
        )

        if self.pretrained:
            self.model.load_state_dict(
                torch.load("path_to_custom_pretrained_weights.pth")
            )

    def forward(self, x):
        return self.model(x)
