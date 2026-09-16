import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from typing import Dict, List, Tuple
from mask import Mask
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distance_est.model import build_dist_model
from inside_pred.model import build_inside_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

class tools_api:
    def __init__(self, dist_model_cfg, inside_model_cfg, resize=(360, 640), mask_IoU_thres=0.3, inside_thres=0.5, img_path=None):
        self.model = build_dist_model(dist_model_cfg)
        self.inside_model = build_inside_model(inside_model_cfg)
        self.resize = resize
        self.mask_IoU_thres = mask_IoU_thres
        self.inside_thres = inside_thres
        self.img_path = img_path
        self.masks = None
        self._rgb_tensor = None
        self._rgb_pil = None
        self._depth_pil = None
        self._resized_mask_cache: Dict[Tuple[int, Tuple[int, int]], np.ndarray] = {}
    
    def update_masks(self, masks: Dict[str, Mask]):
        self.masks = masks
        self._resized_mask_cache.clear()

    def update_image(self, img_path):
        img_path = os.fspath(img_path)
        if img_path == self.img_path and self._rgb_pil is not None:
            return
        self.img_path = img_path
        assert os.path.exists(self.img_path), f"Image path {self.img_path} does not exist."
        self._rgb_tensor = None
        self._rgb_pil = None
        self._depth_pil = None

    def _get_rgb_pil(self) -> Image.Image:
        if self._rgb_pil is None:
            if self.img_path is None:
                raise ValueError("Image path is not set. Call update_image(img_path) first.")
            rgb = Image.open(self.img_path).convert('RGB')
            rgb = F.resize(rgb, self.resize)
            self._rgb_pil = rgb
        return self._rgb_pil

    def _get_depth_pil(self) -> Image.Image:
        if self._depth_pil is None:
            if self.img_path is None:
                raise ValueError("Image path is not set. Call update_image(img_path) first.")
            depth_path = self._depth_path_from_rgb_path(self.img_path)
            if not os.path.exists(depth_path):
                raise FileNotFoundError(f"Depth path {depth_path} does not exist.")
            depth = Image.open(depth_path)
            if depth.mode != 'L':
                depth = depth.convert('L')
            depth = F.resize(depth, self.resize, interpolation=Image.NEAREST)
            self._depth_pil = depth
        return self._depth_pil

    def _depth_path_from_rgb_path(self, rgb_path: str) -> str:
        rgb_path = os.fspath(rgb_path)
        # Expected: .../<split>/images/<frame>.png -> .../<split>/depths/<frame>_depth.png
        depth_path = rgb_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}depths{os.sep}")
        root, ext = os.path.splitext(depth_path)
        return f"{root}_depth{ext}"

    def _get_rgb_tensor(self) -> torch.Tensor:
        if self._rgb_tensor is None:
            rgb = np.asarray(self._get_rgb_pil(), dtype=np.float32) / 255.0
            self._rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(DEVICE)
        return self._rgb_tensor

    def _get_resized_mask(self, mask: Mask) -> np.ndarray:
        key = (id(mask), self.resize)
        cached = self._resized_mask_cache.get(key)
        if cached is not None:
            return cached

        mask_array = mask.decode_mask()
        # Match `DLCV_Final_Project/SpatialAgent/inside_pred/data_loader.py` preprocessing:
        # (mask * 255) uint8 -> resize(BILINEAR) -> ToTensor (== /255.0).
        mask_img = Image.fromarray((mask_array * 255).astype(np.uint8))
        mask_img = F.resize(mask_img, self.resize, interpolation=Image.BILINEAR)
        resized = np.asarray(mask_img, dtype=np.float32) / 255.0
        self._resized_mask_cache[key] = resized
        return resized

    def _mask_bbox_xywh(self, mask_array: np.ndarray) -> tuple[int, int, int, int]:
        rows = np.any(mask_array > 0, axis=1)
        cols = np.any(mask_array > 0, axis=0)

        if not np.any(rows) or not np.any(cols):
            return 0, 0, 0, 0

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

    def _siamese_branch_inputs(self, mask: Mask, crop_size: tuple[int, int] = (224, 224)) -> tuple[torch.Tensor, torch.Tensor]:
        mask_array = mask.decode_mask()
        mask_h, mask_w = mask_array.shape[:2]
        x, y, w, h = self._mask_bbox_xywh(mask_array)

        geo_vector = torch.tensor(
            [
                (x + w / 2) / max(mask_w, 1),
                (y + h / 2) / max(mask_h, 1),
                w / max(mask_w, 1),
                h / max(mask_h, 1),
                (w * h) / max(mask_w * mask_h, 1),
            ],
            dtype=torch.float32,
        )

        rgb_resized = self._get_rgb_pil()
        depth_resized = self._get_depth_pil()
        resized_w, resized_h = rgb_resized.size

        scale_x = resized_w / max(mask_w, 1)
        scale_y = resized_h / max(mask_h, 1)
        x_s = int(x * scale_x)
        y_s = int(y * scale_y)
        w_s = max(1, int(w * scale_x))
        h_s = max(1, int(h * scale_y))

        pad_factor = 0.1
        pad_x = int(w_s * pad_factor)
        pad_y = int(h_s * pad_factor)

        x1 = max(0, x_s - pad_x)
        y1 = max(0, y_s - pad_y)
        x2 = min(resized_w, x_s + w_s + pad_x)
        y2 = min(resized_h, y_s + h_s + pad_y)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, resized_w, resized_h

        rgb_crop = rgb_resized.crop((x1, y1, x2, y2))
        depth_crop = depth_resized.crop((x1, y1, x2, y2))

        rgb_crop = F.resize(rgb_crop, crop_size, interpolation=Image.BILINEAR)
        depth_crop = F.resize(depth_crop, crop_size, interpolation=Image.BILINEAR)

        rgb_tensor = F.to_tensor(rgb_crop)
        depth_tensor = F.to_tensor(depth_crop)
        rgb_tensor = (rgb_tensor - IMAGENET_MEAN) / IMAGENET_STD

        img_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)
        return img_tensor, geo_vector

    def dist(self, mask_1: Mask, mask_2: Mask) -> float:

        if mask_1.object_class.lower() == 'buffer' and self.inside(mask_1, [mask_2]):
            return 0.0
        
        if mask_2.object_class.lower() == 'buffer' and self.inside(mask_2, [mask_1]):
            return 0.0

        img_a, geo_a = self._siamese_branch_inputs(mask_1)
        img_b, geo_b = self._siamese_branch_inputs(mask_2)
        img_a = img_a.to(DEVICE).unsqueeze(0)
        geo_a = geo_a.to(DEVICE).unsqueeze(0)
        img_b = img_b.to(DEVICE).unsqueeze(0)
        geo_b = geo_b.to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            predicted_distance = float(self.model(img_a, geo_a, img_b, geo_b).item())

        predicted_distance = max(0.0, predicted_distance)
        return round(predicted_distance, 2)


    def _centroid(self, mask_array: np.ndarray):
        """Compute the centroid (x, y) of a mask."""
        y_indices, x_indices = np.where(mask_array > 0)
        if len(x_indices) == 0:
            return (0, 0)
        return (np.mean(x_indices), np.mean(y_indices))

    def closest(self, mask_A: Mask, masks: List[Mask]) -> str:
        if not masks:
            raise ValueError("No masks provided to find the closest mask.")

        img_a, geo_a = self._siamese_branch_inputs(mask_A)
        img_a = img_a.to(DEVICE)
        geo_a = geo_a.to(DEVICE)

        img_b_list = []
        geo_b_list = []
        for m in masks:
            img_b, geo_b = self._siamese_branch_inputs(m)
            img_b_list.append(img_b)
            geo_b_list.append(geo_b)

        img_b_batch = torch.stack(img_b_list, dim=0).to(DEVICE)
        geo_b_batch = torch.stack(geo_b_list, dim=0).to(DEVICE)
        img_a_batch = img_a.unsqueeze(0).expand(len(masks), -1, -1, -1)
        geo_a_batch = geo_a.unsqueeze(0).expand(len(masks), -1)

        with torch.no_grad():
            predicted_distances = self.model(img_a_batch, geo_a_batch, img_b_batch, geo_b_batch).squeeze(1)
            predicted_distances = torch.clamp(predicted_distances, min=0.0)

        min_index = int(torch.argmin(predicted_distances).item())
        return masks[min_index].mask_name()


    def is_left(self, mask_A: Mask, mask_B: Mask) -> bool:
        centroid_A = self._centroid(mask_A.decode_mask())
        centroid_B = self._centroid(mask_B.decode_mask())
        return centroid_A[0] < centroid_B[0]

    def is_right(self, mask_A: Mask, mask_B: Mask) -> bool:
        return not self.is_left(mask_A, mask_B)
    
    def mask_IoU(self, mask_A: Mask, mask_B: Mask) -> float:
        maskA_array = mask_A.decode_mask()
        maskB_array = mask_B.decode_mask()
        intersection = np.logical_and(maskA_array, maskB_array).sum()
        union = np.logical_or(maskA_array, maskB_array).sum()
        iou = intersection / union if union > 0 else 0.0
        return iou

    def inside(self, mask_A: Mask, masks: List[Mask]) -> int:
        if not masks:
            return 0
        rgb = self._get_rgb_tensor()

        # Inside model was trained with [RGB(3) + container_mask(1) + obj_mask(1)].
        container_resized = self._get_resized_mask(mask_A)
        container_tensor = torch.from_numpy(container_resized).to(rgb.device).unsqueeze(0)  # 1 x H x W

        obj_batch = torch.stack(
            [torch.from_numpy(self._get_resized_mask(m)) for m in masks],
            dim=0,
        ).to(rgb.device).unsqueeze(1)  # N x 1 x H x W

        rgb_batch = rgb.unsqueeze(0).repeat(len(masks), 1, 1, 1)  # N x 3 x H x W
        container_batch = container_tensor.unsqueeze(0).repeat(len(masks), 1, 1, 1)  # N x 1 x H x W
        batch_tensor = torch.cat([rgb_batch, container_batch, obj_batch], dim=1)  # N x 5 x H x W

        with torch.no_grad():
            # Output: logits, convert to 0/1 using torch.round on sigmoid
            logits = self.inside_model(batch_tensor)
            preds = torch.sigmoid(logits)
            preds = torch.round(preds).long().cpu().numpy()  # 1: inside, 0: outside

        count = int(preds.sum())
        return count


    def most_right(self, masks: List[Mask]) -> int:
        max_x = -1
        rightmost_id = -1
        rightmost_mask = None
        for m in masks:
            centroid = self._centroid(m.decode_mask())
            if centroid[0] > max_x:
                max_x = centroid[0]
                rightmost_id = m.object_id
                rightmost_mask = m
        return rightmost_mask.mask_name()

    def most_left(self, masks: List[Mask]) -> int:
        min_x = float('inf')
        leftmost_id = -1
        leftmost_mask = None
        for m in masks:
            centroid = self._centroid(m.decode_mask())
            if centroid[0] < min_x:
                min_x = centroid[0]
                leftmost_id = m.object_id
                leftmost_mask = m
        return leftmost_mask.mask_name()
    
    def middle(self, masks: List[Mask]) -> str:
        # return the mask that is in the middle of all masks
        if not masks:
            raise ValueError("No masks provided to find the middle mask.")
        assert len(masks) == 3, "The middle function requires exactly 3 masks."
        centroids = []
        for m in masks:
            centroid_x, _ = self._centroid(m.decode_mask())
            centroids.append((centroid_x, m))

        # Sort masks by centroid_x
        centroids.sort(key=lambda x: x[0])

        # Return the mask name of the middle one
        middle_mask = centroids[1][1]
        return middle_mask.mask_name()

    def is_empty(self, transporter_masks: List[Mask]) -> str:
        """
        For each transporter mask, calculate the maximum IoU with all pallet masks.
        Return the transporter mask with the smallest maximum IoU.
        """
        min_max_IoU = float('inf')
        selected_transporter = None

        pallet_masks = [m for m in self.masks.values() if 'pallet' in m.object_class.lower()]

        for transporter in transporter_masks:
            if transporter.object_class.lower() != 'transporter':
                continue

            max_IoU = 0.0
            for pallet in pallet_masks:
                iou = self.mask_IoU(transporter, pallet)
                if iou > max_IoU:
                    max_IoU = iou

            if max_IoU < min_max_IoU:
                min_max_IoU = max_IoU
                selected_transporter = transporter

        if selected_transporter is not None:
            return selected_transporter.mask_name()
        else:
            raise ValueError("No transporter masks found in the provided list.")
