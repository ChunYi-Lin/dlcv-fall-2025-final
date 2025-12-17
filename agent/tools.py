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
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class tools_api:
    def __init__(
        self,
        dist_model_cfg,
        inside_model_cfg,
        small_dist_model_cfg=None,
        resize=(360, 640),
        mask_IoU_thres=0.3,
        inside_thres=0.5,
        cascade_dist_thres=3.0,
        clamp_distance_thres=0.25,
        img_path=None,
        crop_size=(224, 224),
    ):
        self.model = build_dist_model(dist_model_cfg)
        self.inside_model = build_inside_model(inside_model_cfg)
        self.small_dist_model = build_dist_model(small_dist_model_cfg) if small_dist_model_cfg else None
        self.resize = resize
        self.crop_size = crop_size
        self.mask_IoU_thres = mask_IoU_thres
        self.inside_thres = inside_thres
        self.cascade_dist_thres = cascade_dist_thres
        self.clamp_distance_thres = clamp_distance_thres
        self.img_path = img_path
        self.depth_path = None
        self.masks = None
        self._rgb_tensor = None
        self._rgb_pil_resized = None
        self._depth_pil_resized = None
        self._resized_mask_cache: Dict[Tuple[int, Tuple[int, int]], np.ndarray] = {}
        self._dist_input_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    def update_masks(self, masks: Dict[str, Mask]):
        self.masks = masks
        self._resized_mask_cache.clear()
        self._dist_input_cache.clear()

    def update_image(self, img_path):
        img_path = os.fspath(img_path)
        if img_path == self.img_path and self._rgb_tensor is not None:
            return
        self.img_path = img_path
        assert os.path.exists(self.img_path), f"Image path {self.img_path} does not exist."
        self.depth_path = self._infer_depth_path(self.img_path)
        self._rgb_tensor = None
        self._rgb_pil_resized = None
        self._depth_pil_resized = None
        self._dist_input_cache.clear()
        self._get_rgb_tensor()

    def _infer_depth_path(self, rgb_path: str) -> str:
        rgb_dir = os.path.dirname(rgb_path)
        parent_dir = os.path.dirname(rgb_dir)
        depth_dir = os.path.join(parent_dir, 'depths')
        basename = os.path.basename(rgb_path)
        depth_name = basename.replace('.png', '_depth.png')
        return os.path.join(depth_dir, depth_name)

    def _ensure_resized_images_loaded(self) -> None:
        if self._rgb_pil_resized is not None and self._depth_pil_resized is not None:
            return
        if self.img_path is None:
            raise ValueError("Image path is not set. Call update_image(img_path) first.")

        rgb = Image.open(self.img_path).convert('RGB')
        self._rgb_pil_resized = F.resize(rgb, self.resize)

        depth_img = None
        if self.depth_path and os.path.exists(self.depth_path):
            depth_img = Image.open(self.depth_path)
            if depth_img.mode != 'L':
                depth_img = depth_img.convert('L')
        if depth_img is None:
            depth_img = Image.new('L', self._rgb_pil_resized.size, 0)

        self._depth_pil_resized = F.resize(depth_img, self.resize, interpolation=Image.BILINEAR)

    def _get_rgb_tensor(self) -> torch.Tensor:
        if self._rgb_tensor is None:
            self._ensure_resized_images_loaded()
            self._rgb_tensor = F.to_tensor(self._rgb_pil_resized).to(DEVICE)
        return self._rgb_tensor

    def _get_resized_mask(self, mask: Mask) -> np.ndarray:
        key = (id(mask), self.resize)
        cached = self._resized_mask_cache.get(key)
        if cached is not None:
            return cached

        mask_array = mask.decode_mask()
        mask_img = Image.fromarray(mask_array.astype(np.uint8))
        mask_img = F.resize(mask_img, self.resize, interpolation=Image.NEAREST)
        resized = np.asarray(mask_img, dtype=np.float32)
        self._resized_mask_cache[key] = resized
        return resized

    def _bbox_from_mask(self, mask_array: np.ndarray) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask_array > 0)
        if xs.size == 0 or ys.size == 0:
            return 0, 0, 0, 0
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1

    def _prepare_dist_inputs(self, mask: Mask) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = self._dist_input_cache.get(id(mask))
        if cached is not None:
            return cached

        self._ensure_resized_images_loaded()

        mask_array = mask.decode_mask()
        orig_h, orig_w = mask_array.shape
        x, y, w, h = self._bbox_from_mask(mask_array)

        if w <= 0 or h <= 0:
            img_tensor = torch.zeros((4, *self.crop_size), dtype=torch.float32, device=DEVICE)
            geo_vector = torch.zeros((5,), dtype=torch.float32, device=DEVICE)
            self._dist_input_cache[id(mask)] = (img_tensor, geo_vector)
            return img_tensor, geo_vector

        geo_vector = torch.tensor(
            [
                (x + w / 2) / orig_w,
                (y + h / 2) / orig_h,
                w / orig_w,
                h / orig_h,
                (w * h) / (orig_w * orig_h),
            ],
            dtype=torch.float32,
            device=DEVICE,
        )

        resized_w, resized_h = self._rgb_pil_resized.size
        scale_x = resized_w / orig_w
        scale_y = resized_h / orig_h

        x_s = int(round(x * scale_x))
        y_s = int(round(y * scale_y))
        w_s = int(round(w * scale_x))
        h_s = int(round(h * scale_y))

        pad_factor = 0.1
        pad_x = int(round(w_s * pad_factor))
        pad_y = int(round(h_s * pad_factor))

        x1 = max(0, x_s - pad_x)
        y1 = max(0, y_s - pad_y)
        x2 = min(resized_w, x_s + w_s + pad_x)
        y2 = min(resized_h, y_s + h_s + pad_y)

        if x2 <= x1 or y2 <= y1:
            img_tensor = torch.zeros((4, *self.crop_size), dtype=torch.float32, device=DEVICE)
            self._dist_input_cache[id(mask)] = (img_tensor, geo_vector)
            return img_tensor, geo_vector

        rgb_crop = self._rgb_pil_resized.crop((x1, y1, x2, y2))
        depth_crop = self._depth_pil_resized.crop((x1, y1, x2, y2))

        rgb_crop = F.resize(rgb_crop, self.crop_size, interpolation=Image.BILINEAR)
        depth_crop = F.resize(depth_crop, self.crop_size, interpolation=Image.BILINEAR)

        rgb_tensor = F.to_tensor(rgb_crop)
        rgb_tensor = F.normalize(rgb_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        depth_tensor = F.to_tensor(depth_crop)

        img_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0).to(DEVICE)
        self._dist_input_cache[id(mask)] = (img_tensor, geo_vector)
        return img_tensor, geo_vector

    def dist(self, mask_1: Mask, mask_2: Mask) -> float:

        if mask_1.object_class.lower() == 'buffer' and self.inside(mask_1, [mask_2]):
            return 0.0
        
        if mask_2.object_class.lower() == 'buffer' and self.inside(mask_2, [mask_1]):
            return 0.0

        img_a, geo_a = self._prepare_dist_inputs(mask_1)
        img_b, geo_b = self._prepare_dist_inputs(mask_2)

        with torch.inference_mode():
            preds = self.model(
                img_a.unsqueeze(0),
                geo_a.unsqueeze(0),
                img_b.unsqueeze(0),
                geo_b.unsqueeze(0),
            )
            predicted_distance = float(preds.squeeze().item())

        if self.small_dist_model is not None and predicted_distance < self.cascade_dist_thres:
            with torch.inference_mode():
                preds = self.small_dist_model(
                    img_a.unsqueeze(0),
                    geo_a.unsqueeze(0),
                    img_b.unsqueeze(0),
                    geo_b.unsqueeze(0),
                )
                predicted_distance = float(preds.squeeze().item())

        if predicted_distance < self.clamp_distance_thres:
            predicted_distance = 0.0

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
        img_a, geo_a = self._prepare_dist_inputs(mask_A)

        img_bs = []
        geo_bs = []
        for m in masks:
            img_b, geo_b = self._prepare_dist_inputs(m)
            img_bs.append(img_b)
            geo_bs.append(geo_b)

        img_b_batch = torch.stack(img_bs, dim=0)
        geo_b_batch = torch.stack(geo_bs, dim=0)
        img_a_batch = img_a.unsqueeze(0).repeat(len(masks), 1, 1, 1)
        geo_a_batch = geo_a.unsqueeze(0).repeat(len(masks), 1)

        with torch.inference_mode():
            preds = self.model(img_a_batch, geo_a_batch, img_b_batch, geo_b_batch)
            predicted_distances = preds.squeeze(1).detach().cpu().numpy()

        predicted_distances = np.maximum(predicted_distances, 0.0)
        min_index = int(np.argmin(predicted_distances))
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

        # Decode and resize mask_A
        maskA_resized = self._get_resized_mask(mask_A)
        maskA_tensor = torch.from_numpy(maskA_resized).to(rgb.device).unsqueeze(0)

        base = torch.cat([rgb, maskA_tensor], dim=0)  # 4 x H x W
        base_batch = base.unsqueeze(0).expand(len(masks), -1, -1, -1)  # N x 4 x H x W
        maskB_batch = torch.stack(
            [torch.from_numpy(self._get_resized_mask(m)) for m in masks],
            dim=0,
        ).to(rgb.device).unsqueeze(1)  # N x 1 x H x W
        batch_tensor = torch.cat([base_batch, maskB_batch], dim=1)  # N x 5 x H x W

        with torch.no_grad():
            # Output: logits, convert to 0/1 using torch.round on sigmoid
            logits = self.inside_model(batch_tensor)
            probs = torch.sigmoid(logits)
            preds = (probs > self.inside_thres).long().cpu().numpy()  # 1: inside, 0: outside

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
