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
import torch
import torch.nn as nn

# Define LeakyReLU or use ReLU
LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)
# ReLU = nn.ReLU(inplace=True)

class Bottleneck(nn.Module):
    """
    ResNet Bottleneck Block.
    Uses 1x1 Conv (down) -> 3x3 Conv -> 1x1 Conv (up) structure.
    Includes option for downsampling in the shortcut.
    Batch Norm and Activation (LeakyReLU or ReLU) are included.
    Dropout added after the second BN.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = LeakyReLU

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = LeakyReLU

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
        self.stride = stride

        self.dropout_rate = dropout_rate # Store dropout rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=self.dropout_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.dropout_rate > 0:
            out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        self.relu_final = LeakyReLU
        out = self.relu_final(out)

        return out

# Optional: Bottleneck block with Squeeze-and-Excitation (SE) layer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Bottleneck_SE(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.0, se_reduction=16):
        super(Bottleneck_SE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = LeakyReLU

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = LeakyReLU

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.se = SELayer(out_channels * self.expansion, reduction=se_reduction)

        self.downsample = downsample
        self.stride = stride

        self.dropout_rate = dropout_rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=dropout_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out) # Apply SE layer

        if self.dropout_rate > 0:
            out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        self.relu_final = LeakyReLU
        out = self.relu_final(out)

        return out


class MyNetwork_ResNetLarge(nn.Module):
    """
    ResNet architecture using Bottleneck Blocks (e.g., ResNet50, 101, 152)
    designed for ImageNet-scale performance.
    Includes Batch Norm, Leaky ReLU (or ReLU), Dropout in blocks, and initialization.
    Suitable for ~224x224 input images.
    """
    def __init__(self, block, layers, num_classes=1000, dropout_rate=0.0):
        super(MyNetwork_ResNetLarge, self).__init__()
        self.in_channels = 64
        self.dropout_rate = dropout_rate
        self.block = block

        # Initial convolutional layer for ImageNet-scale input (~224x224)
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = LeakyReLU # Or ReLU
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet stages (using Bottleneck blocks)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        # Initialize weights
# --- 가중치 초기화 (수정된 부분) ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # weight와 bias 속성이 None이 아닌 경우에만 초기화 수행
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                 # weight와 bias 속성이 None이 아닌 경우에만 초기화 수행
                if m.weight is not None:
                    nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, self.dropout_rate))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# --- To create a model for ImageNet-scale performance ---

# ResNet-50: A common baseline for ImageNet.
# Highly recommended to use the SE-enhanced version if computational resources allow (which you have).
# model = MyNetwork_ResNetLarge(Bottleneck, [3, 4, 6, 3], num_classes=200, dropout_rate=0.1) # ResNet50
# model = MyNetwork_ResNetLarge(Bottleneck_SE, [3, 4, 6, 3], num_classes=200, dropout_rate=0.1) # SE-ResNet50

# ResNet-101: Deeper, more powerful (in theory). Much higher risk of overfitting from scratch on small data.
# model = MyNetwork_ResNetLarge(Bottleneck, [3, 4, 23, 3], num_classes=200, dropout_rate=0.1) # ResNet101
# model = MyNetwork_ResNetLarge(Bottleneck_SE, [3, 4, 23, 3], num_classes=200, dropout_rate=0.1) # SE-ResNet101

# ResNet-152: Very deep. Extreme risk of overfitting from scratch on small data. Generally not feasible without pre-training.
# model = MyNetwork_ResNetLarge(Bottleneck, [3, 8, 36, 3], num_classes=200, dropout_rate=0.1) # ResNet152
# model = MyNetwork_ResNetLarge(Bottleneck_SE, [3, 8, 36, 3], num_classes=200, dropout_rate=0.1) # SE-ResNet152

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
            self.model = MyNetwork_ResNetLarge(Bottleneck_SE, [3, 4, 6, 3], num_classes=200, dropout_rate=0.1)
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
