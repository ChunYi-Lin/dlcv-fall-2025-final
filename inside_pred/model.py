import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
import pycocotools.mask as mask_utils  # install pycocotools if needed

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_inside_model(model_cfg):
    in_channels = model_cfg.get('input_channels', 5)
    resnet_weights = model_cfg.get('resnet_weights', None)
    model = ResNet50Binary(in_channels=in_channels, resnet_weights=resnet_weights)

    # load model from path
    if 'model_path' in model_cfg and model_cfg['model_path'] is not None:
        model_path = os.fspath(model_cfg['model_path'])
        if not model_path.endswith('.pth'):
            raise ValueError("Model path must end with .pth")
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    model.to(DEVICE)
    return model

class ResNet50Binary(nn.Module):
    def __init__(self, in_channels=5, resnet_weights=None):
        super().__init__()
        self.resnet = models.resnet50(weights=resnet_weights)

        original_conv1_weight = self.resnet.conv1.weight.clone()
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            keep = min(3, in_channels)
            self.resnet.conv1.weight[:, :keep, :, :] = original_conv1_weight[:, :keep, :, :]
            if in_channels > 3:
                extra = in_channels - 3
                mean_w = original_conv1_weight.mean(dim=1, keepdim=True)
                self.resnet.conv1.weight[:, 3:, :, :] = mean_w.repeat(1, extra, 1, 1)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        nn.init.xavier_uniform_(self.resnet.fc.weight)
        nn.init.constant_(self.resnet.fc.bias, 0.0)
        
    def forward(self, x):
        x = self.resnet(x)
        return x.squeeze(1)  # [B] logits
