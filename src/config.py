import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = int(1536/1.7)
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 200
# OPTIMIZER_PARAMS    = {'type': 'AdamW', 'lr': 1e-3}
OPTIMIZER_PARAMS    = {'type': 'AdamW', 'lr': 1e-3, 'weight_decay': 1e-3}

# SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.2}
SCHEDULER_PARAMS = {
    'type': 'CosineAnnealingLR',
    'T_max': NUM_EPOCHS,
    'eta_min': 1e-5,
}

# Dataaset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = int(90/1)
# Augmentation
COLOR_JITTER_PARAMS = {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1} # Add Color Jitter
RANDOM_ERASING_PROB = 0.25 # Add Random Erasing
MIXUP_ALPHA = 0.2 # Add Mixup
CUTMIX_ALPHA = 1.0 # Add Cutmix
IMAGE_ROTATION      = 25
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# Network
MODEL_NAME          = 'resnext101_32x8d'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [0,1,2,3,4,5]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'
# WANDB_DIR = "~/workspace/AUE8088-PA1/wandb"

# AssertionError: Unknown model name: q. Choose one from alexnet, convnext_base, convnext_large, convnext_small, convnext_tiny, deeplabv3_mobilenet_v3_large, deeplabv3_resnet101, deeplabv3_resnet50, densenet121, densenet161, densenet169, densenet201, efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7, efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s, fasterrcnn_mobilenet_v3_large_320_fpn, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, fcn_resnet101, fcn_resnet50, fcos_resnet50_fpn, googlenet, inception_v3, keypointrcnn_resnet50_fpn, lraspp_mobilenet_v3_large, maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2, maxvit_t, mc3_18, mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, mvit_v1_b, mvit_v2_s, quantized_googlenet, quantized_inception_v3, quantized_mobilenet_v2, quantized_mobilenet_v3_large, quantized_resnet18, quantized_resnet50, quantized_resnext101_32x8d, quantized_resnext101_64x4d, quantized_shufflenet_v2_x0_5, quantized_shufflenet_v2_x1_0, quantized_shufflenet_v2_x1_5, quantized_shufflenet_v2_x2_0, r2plus1d_18, r3d_18, raft_large, raft_small, regnet_x_16gf, regnet_x_1_6gf, regnet_x_32gf, regnet_x_3_2gf, regnet_x_400mf, regnet_x_800mf, regnet_x_8gf, regnet_y_128gf, regnet_y_16gf, regnet_y_1_6gf, regnet_y_32gf, regnet_y_3_2gf, regnet_y_400mf, regnet_y_800mf, regnet_y_8gf, resnet101, resnet152, resnet18, resnet34, resnet50, resnext101_32x8d, resnext101_64x4d, resnext50_32x4d, retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2, s3d, shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0, squeezenet1_0, squeezenet1_1, ssd300_vgg16, ssdlite320_mobilenet_v3_large, swin3d_b, swin3d_s, swin3d_t, swin_b, swin_s, swin_t, swin_v2_b, swin_v2_s, swin_v2_t, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn, vit_b_16, vit_b_32, vit_h_14, vit_l_16, vit_l_32, wide_resnet101_2, wide_resnet50_2