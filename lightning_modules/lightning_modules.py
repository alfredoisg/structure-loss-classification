import pytorch_lightning as pl

import torch
import torch.nn as nn
import torchmetrics

from models.models import LeNet5


class LitModelBase(pl.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float) -> None:
        super().__init__()

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def _common_step(self, batch, batch_idx, stage=None):
        inputs, labels = batch
        predictions = self(inputs)
        loss = self.loss_fn(inputs, predictions)

        acc = self.accuracy(predictions, labels)
        f1 = self.f1_score(predictions, labels)

        if stage:
            self.log_dict(
                {
                    f"{stage}_loss": loss,
                    f"{stage}_accuracy": acc,
                    f"{stage}_f1_score": f1,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def training_step(self, batch, batch_idx):
        return _common_step(self, batch, batch_idx, stage=None)

    def validation_step(self, batch, batch_idx):
        return _common_step(self, batch, batch_idx, stage="val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LitLeNet5(LitModelBase):
    def __init__(self, num_classes: int, learning_rate: float) -> None:
        super().__init__(num_classes, learning_rate)

        self.model = LeNet5()

    def forward(self, x):
        out = self.model(x)
        return out
