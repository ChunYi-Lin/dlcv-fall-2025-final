import torch
import torch.nn as nn
import torchvision.models as models

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_dist_model(model_cfg: dict) -> nn.Module:
    arch = model_cfg.get('arch', 'siamese')
    if arch != 'siamese':
        raise ValueError(f"Unsupported distance model arch: {arch!r} (expected 'siamese').")

    model_path = model_cfg.get('model_path')
    if not model_path:
        raise ValueError("Missing required distance model config key: 'model_path'.")
    if not str(model_path).endswith('.pth'):
        raise ValueError("Model path must end with .pth")

    model = GeoAwareSiameseNet(
        visual_dim=model_cfg.get('visual_dim', 2048),
        geo_input_dim=model_cfg.get('geo_input_dim', 5),
        geo_embed_dim=model_cfg.get('geo_embed_dim', 256),
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model


class GeoAwareSiameseNet(nn.Module):
    def __init__(self, visual_dim: int = 2048, geo_input_dim: int = 5, geo_embed_dim: int = 256):
        super().__init__()

        base_model = models.resnet50(weights=None)
        original_first_layer = base_model.conv1

        new_first_layer = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_first_layer.weight[:, :3, :, :] = original_first_layer.weight
            new_first_layer.weight[:, 3:4, :, :] = torch.mean(original_first_layer.weight, dim=1, keepdim=True)

        base_model.conv1 = new_first_layer
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
        vis_feat = self.visual_backbone(img_crop).flatten(1)
        geo_feat = self.geo_encoder(geo_vector)
        return torch.cat([vis_feat, geo_feat], dim=1)

    def forward(self, img_a: torch.Tensor, geo_a: torch.Tensor, img_b: torch.Tensor, geo_b: torch.Tensor) -> torch.Tensor:
        feat_a = self.forward_one_branch(img_a, geo_a)
        feat_b = self.forward_one_branch(img_b, geo_b)

        diff = torch.abs(feat_a - feat_b)
        prod = feat_a * feat_b

        interaction_feat = torch.cat([feat_a, feat_b, diff, prod], dim=1)
        return self.regressor(interaction_feat)
