"""
Baseline SAM 2 Tracker

This module defines the pure SAM 2 tracking logic with raw video predictor chunking
(no Kalman Filter or Memory Bank).
"""

import torch
import numpy as np
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SAM2Tracker:
    """SAM 2 baseline tracker with chunked inference (Pure SAM2)."""

    def __init__(self, sam2_config: str, sam2_checkpoint: str):
        self.device = DEVICE
        print(f"Loading SAM 2 model on {self.device}...")
        self.predictor = build_sam2_video_predictor(
            sam2_config,
            sam2_checkpoint,
            device=self.device,
        )
        print("  ✓ Baseline Model loaded")

    def track(self, img_dir: str, init_bbox: List[int],
              num_frames: int, chunk_size: int = 100) -> Tuple[List[List[int]], List[float]]:
        """Track object using chunked SAM 2 inference.

        Args:
            img_dir: Directory of JPEG frames
            init_bbox: [x, y, w, h] from ground truth
            num_frames: Total frames available
            chunk_size: Frames per chunk

        Returns:
            List of [x, y, w, h] predictions per frame
        """
        x, y, w, h = init_bbox
        box_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)

        # Get frame files
        img_path = Path(img_dir)
        frame_files = sorted(img_path.glob("*.jpg"))
        if not frame_files:
            frame_files = sorted(img_path.glob("*.png"))

        # Split into chunks
        chunks = []
        for start in range(0, len(frame_files), chunk_size):
            end = min(start + chunk_size, len(frame_files))
            chunks.append(frame_files[start:end])

        predictions = []
        occlusion_scores = []
        carry_mask = None

        for ci, chunk_files in enumerate(chunks):
            tmp_dir = tempfile.mkdtemp(prefix=f"sam2_c{ci}_")
            try:
                for i, src in enumerate(chunk_files):
                    dst = Path(tmp_dir) / f"{i:06d}.jpg"
                    os.symlink(str(src.resolve()), str(dst))

                with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                    state = self.predictor.init_state(
                        video_path=tmp_dir,
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=False,
                        async_loading_frames=True,
                    )

                    if ci == 0:
                        _, _, masks = self.predictor.add_new_points_or_box(
                            inference_state=state, frame_idx=0, obj_id=1, box=box_xyxy)
                    else:
                        _, _, masks = self.predictor.add_new_mask(
                            inference_state=state, frame_idx=0, obj_id=1, mask=carry_mask)

                    mask_np = (masks[0] > 0.0).cpu().numpy().squeeze()
                    predictions.append(self._mask_to_bbox(mask_np))
                    occlusion_scores.append(1.0 - torch.sigmoid(masks[0].max()).item())

                    last_masks = masks
                    for frame_idx, _, masks in tqdm(self.predictor.propagate_in_video(state), desc=f"Propagating Chunk {ci+1}/{len(chunks)}", leave=False):
                        if frame_idx == 0:
                            continue
                        mask_np = (masks[0] > 0.0).cpu().numpy().squeeze()
                        predictions.append(self._mask_to_bbox(mask_np))
                        occlusion_scores.append(1.0 - torch.sigmoid(masks[0].max()).item())
                        last_masks = masks

                    carry_mask = (last_masks[0] > 0.0).cpu().numpy().squeeze()
                    self.predictor.reset_state(state)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            if self.device == "cuda":
                torch.cuda.empty_cache()

        return predictions, occlusion_scores

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> List[int]:
        if mask.sum() == 0:
            return [0, 0, 10, 10]
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
