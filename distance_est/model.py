import os
import torchvision.models as models
import torch.nn as nn
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_dist_model(model_cfg):
    """
    Build a distance model based on the provided configuration.
    
    Args:
        model_cfg (dict): Configuration dictionary containing model parameters.
    
    Returns:
        nn.Module: The constructed distance model.
    """
    model_arch = model_cfg.get('arch', 'siamese_geo')
    input_channels = model_cfg.get('input_channels', 5)
    backbone = model_cfg.get('backbone', 'resnet50')

    if model_arch == 'resnet_regressor':
        model = ResNetDistanceRegressor(input_channels=input_channels, backbone=backbone, pretrained=False)
    elif model_arch == 'siamese_geo':
        model = GeoAwareSiameseNet()
    else:
        raise ValueError(f"Unknown distance model arch: {model_arch}")

    # Load model from path (recommended for inference).
    if 'model_path' in model_cfg and model_cfg['model_path'] is not None:
        model_path = os.fspath(model_cfg['model_path'])
        if not model_path.endswith('.pth'):
            raise ValueError("Model path must end with .pth")
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(DEVICE)
    return model


class ResNetDistanceRegressor(nn.Module):
    def __init__(self, input_channels=5, backbone='resnet50', pretrained=False):
        super().__init__()
        self.resnet = getattr(models, backbone)(weights='IMAGENET1K_V1' if pretrained else None)

        # Replace first conv layer to accept custom number of input channels
        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(input_channels, old_conv.out_channels,
                                      kernel_size=old_conv.kernel_size,
                                      stride=old_conv.stride,
                                      padding=old_conv.padding,
                                      bias=old_conv.bias is not None)

        # Copy weights if input_channels == 3
        if pretrained and input_channels == 6:
            with torch.no_grad():
                self.resnet.conv1.weight[:, :3] = old_conv.weight
                self.resnet.conv1.weight[:, 3:] = old_conv.weight.clone()

        num_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_feats, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.resnet(x).squeeze(1)  # Output shape: (B,)


class GeoAwareSiameseNet(nn.Module):
    """
    Siamese distance estimation model (RGB-D local crops + geometry embedding).
    This matches the architecture used by `DLCV_Final_Project/modelY.py`.
    """
    def __init__(self, visual_dim: int = 2048, geo_input_dim: int = 5, geo_embed_dim: int = 256):
        super().__init__()

        base_model = models.resnet50(weights=None)
        base_model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.visual_backbone = nn.Sequential(*list(base_model.children())[:-1])

        self.geo_encoder = nn.Sequential(
            nn.Linear(geo_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, geo_embed_dim),
            nn.LayerNorm(geo_embed_dim),
        )

        combined_dim = (visual_dim + geo_embed_dim) * 4
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward_one_branch(self, img_crop: torch.Tensor, geo_vector: torch.Tensor) -> torch.Tensor:
        vis_feat = self.visual_backbone(img_crop).view(img_crop.size(0), -1)
        geo_feat = self.geo_encoder(geo_vector)
        return torch.cat([vis_feat, geo_feat], dim=1)

    def forward(
        self,
        img_a: torch.Tensor,
        geo_a: torch.Tensor,
        img_b: torch.Tensor,
        geo_b: torch.Tensor,
    ) -> torch.Tensor:
        feat_a = self.forward_one_branch(img_a, geo_a)
        feat_b = self.forward_one_branch(img_b, geo_b)

        diff = torch.abs(feat_a - feat_b)
        prod = feat_a * feat_b
        interaction_feat = torch.cat([feat_a, feat_b, diff, prod], dim=1)
        return self.regressor(interaction_feat)
