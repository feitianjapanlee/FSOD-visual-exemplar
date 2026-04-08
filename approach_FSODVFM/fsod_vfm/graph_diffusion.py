import torch
import torch.nn.functional as F
from torch import nn


class GraphDiffusion(nn.Module):
    def __init__(self, num_steps: int = 30, temperature: float = 0.1):
        super().__init__()
        self.num_steps = num_steps
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
        initial_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (N, D) - normalized feature vectors for proposals
            masks: (N, H, W) - binary masks for each proposal
            initial_scores: (N,) - initial confidence scores
        Returns:
            refined_scores: (N,) - scores after graph diffusion
        """
        n = features.shape[0]
        if n == 0:
            return initial_scores

        edge_weights = self._compute_edge_weights(masks)
        similarity_matrix = self._compute_similarity_matrix(features)

        refined_scores = initial_scores.clone()
        alpha = 0.5

        for _ in range(self.num_steps):
            aggregated = torch.zeros_like(refined_scores)
            for i in range(n):
                w = edge_weights[i]
                if w.sum() > 0:
                    s = similarity_matrix[i] * w
                    s = s / (self.temperature * w.sum())
                    weights = F.softmax(s, dim=0)
                    aggregated[i] = (weights * refined_scores).sum()

            refined_scores = alpha * aggregated + (1 - alpha) * initial_scores

        return refined_scores

    def _compute_edge_weights(self, masks: torch.Tensor) -> torch.Tensor:
        n, h, w = masks.shape
        masks_flat = masks.flatten(1)
        intersection = torch.mm(masks_flat, masks_flat.T)
        mask_areas = masks_flat.sum(dim=1, keepdim=True)
        overlap_ratio = intersection / (mask_areas + 1e-8)
        return overlap_ratio

    def _compute_similarity_matrix(self, features: torch.Tensor) -> torch.Tensor:
        return torch.mm(features, features.T)


class MaskedRoIPooling(nn.Module):
    def __init__(self, output_size: int = 7):
        super().__init__()
        self.output_size = output_size

    def forward(
        self,
        image_features: torch.Tensor,
        masks: torch.Tensor,
        boxes: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            image_features: (B, C, H', W') - feature map from vision backbone
            masks: (N, H, W) - binary masks for each proposal
            boxes: (N, 4) - bounding boxes [x1, y1, x2, y2] in image coordinates
            image_size: (height, width) - original image size
        Returns:
            pooled_features: (N, C, output_size, output_size)
        """
        n = boxes.shape[0]
        if n == 0:
            return torch.empty(
                0,
                image_features.shape[1],
                self.output_size,
                self.output_size,
                device=image_features.device,
                dtype=image_features.dtype,
            )

        h_img, w_img = image_size
        h_feat, w_feat = image_features.shape[2:]

        x_scale = w_feat / w_img
        y_scale = h_feat / h_img

        box_features = []
        for i in range(n):
            x1, y1, x2, y2 = boxes[i].tolist()
            x1_f = x1 * x_scale
            y1_f = y1 * y_scale
            x2_f = x2 * x_scale
            y2_f = y2 * y_scale

            x1_f = max(0, min(x1_f, w_feat - 1))
            y1_f = max(0, min(y1_f, h_feat - 1))
            x2_f = max(x1_f + 1, min(x2_f, w_feat))
            y2_f = max(y1_f + 1, min(y2_f, h_feat))

            feat_patch = image_features[
                0, :, int(y1_f) : int(y2_f), int(x1_f) : int(x2_f)
            ]
            mask_patch = masks[i, int(y1_f) : int(y2_f), int(x1_f) : int(x2_f)]

            if feat_patch.numel() == 0:
                pooled = F.adaptive_avg_pool2d(
                    image_features[
                        0:1, :, int(y1_f) : int(y2_f), int(x1_f) : int(x2_f)
                    ],
                    (self.output_size, self.output_size),
                )
            else:
                mask_patch_up = (
                    F.interpolate(
                        mask_patch.unsqueeze(0).unsqueeze(0).float(),
                        size=feat_patch.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

                weighted_feat = feat_patch * mask_patch_up
                pooled = F.adaptive_avg_pool2d(
                    weighted_feat.unsqueeze(0), (self.output_size, self.output_size)
                )

            box_features.append(pooled)

        return torch.cat(box_features, dim=0)

    def pooled_to_vector(self, pooled: torch.Tensor) -> torch.Tensor:
        return pooled.flatten(1)
