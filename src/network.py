# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch

# Custom packages
from src.metric import MyAccuracy, MyF1Score
import src.config as cfg
from src.util import show_setting


# [TODO: Optional] Rewrite this class if you want
LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)

class MyNetwork(nn.Module):

    def __init__(self, num_classes: int = cfg.NUM_CLASSES):
        super(MyNetwork, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, stride=4, padding=1)
        self.bn1a = nn.BatchNorm2d(64)
        self.lkrelu1a = LeakyReLU

        self.conv1b = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(96)
        self.lkrelu1b = LeakyReLU 

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.25) 

        self.conv2 = nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.lkrelu2 = LeakyReLU 

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout2 = nn.Dropout2d(p=0.25) 

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.lkrelu3 = LeakyReLU 

        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(384)
        self.lkrelu4 = LeakyReLU 

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.lkrelu5 = LeakyReLU

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout3 = nn.Dropout2d(p=0.25) 

        self.skip1_proj = nn.Conv2d(3, 96, kernel_size=1, stride=4)

        self.skip2_proj = nn.Conv2d(96, 256, kernel_size=1, stride=1)

        self.skip3_proj = nn.Conv2d(256, 384, kernel_size=1, stride=1)


        self.skip4_proj = nn.Identity()

        self.skip5_proj = nn.Conv2d(384, 256, kernel_size=1, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) 

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            LeakyReLU,
            nn.Dropout(p=0.5), 
            nn.Linear(4096, 4096),
            LeakyReLU,
            nn.Linear(4096, num_classes), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = self.skip1_proj(x) 
        out = self.conv1a(x)
        out = self.bn1a(out)
        out = self.lkrelu1a(out)

        out = self.conv1b(out)
        out = self.bn1b(out)
        out += identity 
        out = self.lkrelu1b(out)

        out = self.pool1(out)
        out = self.dropout1(out)
        x = out 

        identity = self.skip2_proj(x)
        out = self.conv2(x)
        out = self.bn2(out)
        out += identity
        out = self.lkrelu2(out)

        out = self.pool2(out)
        out = self.dropout2(out)
        x = out 

        identity = self.skip3_proj(x)
        out = self.conv3(x)
        out = self.bn3(out)
        out += identity
        out = self.lkrelu3(out)
        x = out 

        identity = self.skip4_proj(x)
        out = self.conv4(x)
        out = self.bn4(out)
        out += identity
        out = self.lkrelu4(out)
        x = out 

        identity = self.skip5_proj(x)
        out = self.conv5(x)
        out = self.bn5(out)
        out += identity
        out = self.lkrelu5(out)

        out = self.pool3(out)
        out = self.dropout3(out)
        x = out 

        x = self.avgpool(x)
        x = torch.flatten(x, 1) 
        x = self.classifier(x) 

        return x

class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()

        # Network
        if model_name == 'MyNetwork':  
            self.model = MyNetwork()
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        self.accuracy = MyAccuracy()
        self.F1_score = MyF1Score(num_classes, 'macro')

        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1_score = self.F1_score(scores, y)
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy, 'f1_score/train': f1_score},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1_score = self.F1_score(scores, y)
        self.log_dict({'loss/val': loss, 'accuracy/val': accuracy, 'f1_score/val': f1_score},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
