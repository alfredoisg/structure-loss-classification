
import torch
import torch.nn as nn

import pytorch_lightning as pl
import torchmetrics

from torchvision import models

def conv_layer(in_channels, out_channels):
            layer = nn.Sequential(
                nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=5,
                stride=1,
                padding=0
                ),
                nn.BatchNorm2d(
                num_features=out_channels
                ),
                nn.ReLU()
            )
            return layer
## output_size = (input_size - kernel_size + 2*padding)/stride + 1

def fc_layer(in_features, out_features):

            layer = nn.Linear(
                in_features=in_features,
                out_features=out_features
            )

            nn.init.kaiming_normal_(layer.weight,
                                    nonlinearity='relu')
            return layer





class LitLeNet5(pl.LightningModule):
    def __init__(
        self,  params: dict,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()


        self.num_classes = self.hparams.params['model_params']['num_classes']
        self.size_layer_1 = self.hparams.params['model_params']['size_layer_1']
        self.size_layer_2 = self.hparams.params['model_params']['size_layer_2']
        self.learning_rate = self.hparams.params['learning_rate']

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)

        self.cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)

        self.stored_confusion_matrix = None

        self.loss_fn = nn.CrossEntropyLoss()



        convStack = nn.Sequential(
            conv_layer(3, 6), ## 244x244 --> 240x240
            nn.MaxPool2d(2,2), ## 240x240 --> 120x120
            conv_layer(6, 16), ## 120x120 --> 117x117
            nn.MaxPool2d(2, 2), ## 122x122 --> 58x58
            conv_layer(16, 120) ## 58x58 --> 54x54
        )

        fullyConStack = nn.Sequential(
            nn.Flatten(),
            fc_layer(54*54*120, self.size_layer_1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(self.size_layer_1, self.size_layer_2),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(self.size_layer_2, self.num_classes)
        )


        self.convStack = convStack
        self.fullyConStack = fullyConStack

    def forward(self, x):
        x = self.convStack(x)
        x = self.fullyConStack(x)
        return x


    def _common_step(self, batch, batch_idx):
                            # data, count
        inputs, labels = batch
        predictions = self.forward(inputs) # predictions = model(inputs)
        loss = self.loss_fn(predictions, labels)
        return loss, predictions, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, predictions, labels = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(predictions, labels)
        f1_score = self.f1_score(predictions, labels)


        self.log_dict({'val_loss': loss,
                       'val_accuracy': accuracy,
                       'val_f1_score': f1_score},
                       on_step=False,
                       on_epoch=True,
                       prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        _, prediction, _ = self._common_step(batch, batch_idx)
        preds = torch.argmax(prediction, dim=1)
        return preds


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LitVGG16(pl.LightningModule):
    def __init__(
        self,  params: dict,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()


        self.num_classes = self.hparams.params['model_params']['num_classes']
        self.size_layer_1 = self.hparams.params['model_params']['size_layer_1']
        self.size_layer_2 = self.hparams.params['model_params']['size_layer_2']
        self.size_layer_3 = self.hparams.params['model_params']['size_layer_3']
        
        self.learning_rate = self.hparams.params['learning_rate']

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)

        self.cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)

        self.stored_confusion_matrix = None

        self.loss_fn = nn.CrossEntropyLoss()



        convStack = nn.Sequential(
            # C64-C64-P2
            conv_layer(3, 64),
            conv_layer(64, 64),
            nn.MaxPool2d(2, 2),  # 244x244 --> 122x122
            # C128-C128-P2
            conv_layer(64, 128),
            conv_layer(128, 128),
            nn.MaxPool2d(2, 2),  # 122x122 --> 61x61
            # C256-C256-C256-P2
            conv_layer(128, 256),
            conv_layer(256, 256),
            conv_layer(256, 256),
            conv_layer(256, 256),
            nn.MaxPool2d(2, 2),  # 61x61 --> 30x30
            # C512-C512-C512-P2
            conv_layer(256, 512),
            conv_layer(512, 512),
            conv_layer(512, 512),
            nn.MaxPool2d(2, 2),  # 30x30 --> 15x15
            # C512-C512-C512-P2
            conv_layer(512, 512),
            conv_layer(512, 512),
            conv_layer(512, 512),
            nn.MaxPool2d(2, 2),  # 15x15 --> 7x7
        )
        
        fullyConStack = nn.Sequential(
            nn.Flatten(),
            # F4096-F4096-F1000
            fc_layer(512 * 7 * 7, self.size_layer_1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            fc_layer(self.size_layer_1, self.size_layer_2),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            fc_layer(self.size_layer_2, self.size_layer_3),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            fc_layer(self.size_layer_3, self.num_classes),
        )


        self.convStack = convStack
        self.fullyConStack = fullyConStack

    def forward(self, x):
        x = self.convStack(x)
        x = self.fullyConStack(x)
        return x


    def _common_step(self, batch, batch_idx):
                            # data, count
        inputs, labels = batch
        predictions = self.forward(inputs) # predictions = model(inputs)
        loss = self.loss_fn(predictions, labels)
        return loss, predictions, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, predictions, labels = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(predictions, labels)
        f1_score = self.f1_score(predictions, labels)


        self.log_dict({'val_loss': loss,
                       'val_accuracy': accuracy,
                       'val_f1_score': f1_score},
                       on_step=False,
                       on_epoch=True,
                       prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        _, prediction, _ = self._common_step(batch, batch_idx)
        preds = torch.argmax(prediction, dim=1)
        return preds


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    

class LitResNet18(pl.LightningModule):
    def __init__(
        self,  params: dict,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()


        self.num_classes = self.hparams.params['model_params']['num_classes']
        self.size_layer_1 = self.hparams.params['model_params']['size_layer_1']
        self.learning_rate = self.hparams.params['learning_rate']
        self.pretrained = self.hparams.params['pretrained']

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)

        self.cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)

        self.stored_confusion_matrix = None

        self.loss_fn = nn.CrossEntropyLoss()
        
        
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes)
        
        
        if self.pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True



    def forward(self, x):
        x = self.model(x)
        return x


    def _common_step(self, batch, batch_idx):
                            # data, count
        inputs, labels = batch
        predictions = self.forward(inputs) # predictions = model(inputs)
        loss = self.loss_fn(predictions, labels)
        return loss, predictions, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, predictions, labels = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(predictions, labels)
        f1_score = self.f1_score(predictions, labels)


        self.log_dict({'val_loss': loss,
                       'val_accuracy': accuracy,
                       'val_f1_score': f1_score},
                       on_step=False,
                       on_epoch=True,
                       prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        _, prediction, _ = self._common_step(batch, batch_idx)
        preds = torch.argmax(prediction, dim=1)
        return preds


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
