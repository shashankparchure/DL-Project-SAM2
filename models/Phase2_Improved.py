"""
Phase 2 Improved: HiM2SAM-Inspired Motion-Aware SAM 2 Tracker

Key Fixes for Occlusion Handling:
1. NO PREDICTION during occlusion (return None/empty)
2. Template matching from memory bank when object reappears
3. Extended memory bank stores high-conf templates for re-detection
"""

import numpy as np
import torch
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from filterpy.kalman import KalmanFilter as FilterPyKalman
from enum import Enum

from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TrackingState(Enum):
    VISIBLE = "visible"
    UNCERTAIN = "uncertain"
    OCCLUDED = "occluded"
    LOST = "lost"


class HierarchicalKalmanFilter:
    """Kalman Filter for motion prediction. State: [cx, cy, w, h, vx, vy, vw, vh]"""

    def __init__(self):
        self.kf = FilterPyKalman(dim_x=8, dim_z=4)
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],
            [0, 1, 0, 0, 0,  dt, 0,  0],
            [0, 0, 1, 0, 0,  0,  dt, 0],
            [0, 0, 0, 1, 0,  0,  0,  dt],
            [0, 0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0, 0,  0,  0,  1],
        ], dtype=np.float32)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        self.kf.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1]).astype(np.float32)
        self.kf.R = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        self.kf.P *= 1000.0
        self.initialized = False

    def initialize(self, bbox: np.ndarray):
        self.kf.x = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0, 0],
                            dtype=np.float32).reshape(-1, 1)
        self.initialized = True

    def predict(self) -> np.ndarray:
        if not self.initialized:
            return np.array([0, 0, 0, 0], dtype=np.float32)
        self.kf.predict()
        return self.kf.x[:4].flatten()

    def update(self, bbox: np.ndarray):
        if not self.initialized:
            return
        self.kf.update(np.array(bbox, dtype=np.float32).reshape(-1, 1))

    def reset(self):
        self.kf.x = np.zeros((8, 1), dtype=np.float32)
        self.kf.P = np.eye(8, dtype=np.float32) * 1000.0
        self.initialized = False


class ExtendedMemoryBank:
    """
    Extended Memory Bank for storing high-confidence templates.
    Used for re-detection after long occlusions.
    """

    def __init__(self, max_templates: int = 10):
        self.templates = []  # List of high-conf templates
        self.max_templates = max_templates
        self.initial_template = None

    def set_initial(self, bbox: np.ndarray, mask: np.ndarray, features: Optional[np.ndarray] = None):
        """Store initial template - most important for re-detection."""
        self.initial_template = {
            "bbox": bbox.copy(),
            "mask": mask.copy() if mask is not None else None,
            "confidence": 1.0,
            "frame_idx": 0,
        }

    def add_template(self, frame_idx: int, bbox: np.ndarray, mask: np.ndarray,
                     confidence: float) -> bool:
        """Add high-confidence template for future re-detection."""
        if confidence < 0.8:
            return False

        # Check size validity (not too small, not too large)
        if bbox[2] < 10 or bbox[3] < 10:
            return False

        template = {
            "bbox": bbox.copy(),
            "mask": mask.copy() if mask is not None else None,
            "confidence": confidence,
            "frame_idx": frame_idx,
        }

        # Check diversity
        if self._is_diverse(bbox):
            self.templates.append(template)
            # Keep best templates
            if len(self.templates) > self.max_templates:
                # Remove lowest confidence
                min_idx = min(range(len(self.templates)),
                            key=lambda i: self.templates[i]["confidence"])
                self.templates.pop(min_idx)
            return True
        return False

    def _is_diverse(self, bbox: np.ndarray, threshold: float = 0.8) -> bool:
        """Check if bbox is different enough from stored templates."""
        for t in self.templates:
            iou = self._compute_iou(bbox, t["bbox"])
            if iou > threshold:
                return False
        return True

    def get_reference_bbox(self) -> Optional[np.ndarray]:
        """Get best reference bbox for re-detection (prefer initial template)."""
        if self.initial_template:
            return self.initial_template["bbox"].copy()
        if self.templates:
            best = max(self.templates, key=lambda x: x["confidence"])
            return best["bbox"].copy()
        return None

    def get_reference_size(self) -> Tuple[float, float]:
        """Get expected object size (w, h) from templates."""
        sizes = []
        if self.initial_template:
            sizes.append((self.initial_template["bbox"][2], self.initial_template["bbox"][3]))
        for t in self.templates:
            sizes.append((t["bbox"][2], t["bbox"][3]))

        if sizes:
            avg_w = np.mean([s[0] for s in sizes])
            avg_h = np.mean([s[1] for s in sizes])
            return avg_w, avg_h
        return 50, 50  # Default

    def validate_detection(self, bbox: np.ndarray, confidence: float) -> Tuple[bool, float]:
        """
        Validate a detection against stored templates.
        Returns (is_valid, boosted_confidence)
        """
        if bbox[2] <= 0 or bbox[3] <= 0:
            return False, 0.0

        # Get expected size
        exp_w, exp_h = self.get_reference_size()

        # Check size consistency (within 3x range)
        size_ratio_w = bbox[2] / exp_w if exp_w > 0 else 1
        size_ratio_h = bbox[3] / exp_h if exp_h > 0 else 1

        if size_ratio_w < 0.3 or size_ratio_w > 3.0:
            return False, confidence * 0.5
        if size_ratio_h < 0.3 or size_ratio_h > 3.0:
            return False, confidence * 0.5

        # Size is reasonable - boost confidence slightly
        size_score = 1.0 - abs(1.0 - (size_ratio_w + size_ratio_h) / 2) * 0.3
        boosted = min(1.0, confidence * (1 + size_score * 0.2))

        return True, boosted

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        b1 = [box1[0] - box1[2]/2, box1[1] - box1[3]/2,
              box1[0] + box1[2]/2, box1[1] + box1[3]/2]
        b2 = [box2[0] - box2[2]/2, box2[1] - box2[3]/2,
              box2[0] + box2[2]/2, box2[1] + box2[3]/2]
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1, area2 = box1[2] * box1[3], box2[2] * box2[3]
        return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0.0

    def clear(self):
        self.templates = []
        self.initial_template = None


class SAM2KalmanTracker:
    """
    HiM2SAM-Inspired Tracker with proper occlusion handling.

    Key behavior:
    - VISIBLE: Normal tracking, store templates
    - OCCLUDED/LOST: Return NO PREDICTION (None), don't output garbage
    - RE-APPEAR: Use memory bank templates to validate and boost detection
    """

    def __init__(self, mode: str = "kalman",
                 confidence_threshold: float = 0.7,
                 occlusion_threshold: float = 0.3,
                 lost_threshold: float = 0.15,
                 recovery_frames: int = 5):

        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.occlusion_threshold = occlusion_threshold
        self.lost_threshold = lost_threshold
        self.recovery_frames = recovery_frames

        # Load SAM 2
        project_root = Path(__file__).resolve().parent.parent
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        checkpoint = str(project_root / "checkpoints" / "sam2.1_hiera_small.pt")
        if not Path(checkpoint).exists():
            checkpoint = str(project_root / "models" / "sam2.1_hiera_small.pt")

        print(f"Loading SAM 2 on {DEVICE}...")
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=DEVICE)
        print("  ✓ Phase 2 Improved loaded")

        # Components
        self.kalman = HierarchicalKalmanFilter()
        self.memory_bank = ExtendedMemoryBank(max_templates=10)

        # State
        self.tracking_state = TrackingState.VISIBLE
        self.low_conf_count = 0
        self.last_valid_bbox = None
        self.init_bbox_xyxy = None
        self.frames_since_visible = 0

    def track(self, img_dir: str, init_bbox: List[int], num_frames: int,
              chunk_size: int = 200) -> Tuple[List[Optional[List[int]]], List[float]]:
        """
        Track with proper occlusion handling.
        Returns None for frames where object is not visible.
        """
        self._reset()

        x, y, w, h = init_bbox
        init_bbox_center = np.array([x + w/2, y + h/2, w, h], dtype=np.float32)
        self.init_bbox_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)

        self.kalman.initialize(init_bbox_center)
        self.last_valid_bbox = init_bbox_center.copy()

        img_path = Path(img_dir)
        frame_files = sorted(img_path.glob("*.jpg"))
        if not frame_files:
            frame_files = sorted(img_path.glob("*.png"))
        frame_files = frame_files[:num_frames]

        chunks = []
        for start in range(0, len(frame_files), chunk_size):
            end = min(start + chunk_size, len(frame_files))
            chunks.append((start, frame_files[start:end]))

        predictions = []
        occlusion_scores = []
        carry_mask = None

        for ci, (chunk_start, chunk_files) in enumerate(chunks):
            tmp_dir = tempfile.mkdtemp(prefix=f"him2sam_c{ci}_")
            try:
                for i, src in enumerate(chunk_files):
                    dst = Path(tmp_dir) / f"{i:06d}.jpg"
                    os.symlink(str(src.resolve()), str(dst))

                with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                    state = self.predictor.init_state(
                        video_path=tmp_dir,
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=False,
                        async_loading_frames=True,
                    )

                    if ci == 0:
                        _, _, masks = self.predictor.add_new_points_or_box(
                            inference_state=state, frame_idx=0, obj_id=1,
                            box=self.init_bbox_xyxy
                        )
                    else:
                        if carry_mask is not None and carry_mask.sum() > 0:
                            _, _, masks = self.predictor.add_new_mask(
                                inference_state=state, frame_idx=0, obj_id=1,
                                mask=carry_mask
                            )
                        else:
                            # Re-init with initial bbox if mask is empty
                            _, _, masks = self.predictor.add_new_points_or_box(
                                inference_state=state, frame_idx=0, obj_id=1,
                                box=self.init_bbox_xyxy
                            )

                    mask_np = (masks[0] > 0.0).cpu().numpy().squeeze()
                    confidence = float(torch.sigmoid(masks[0].max()).item())

                    pred, occ = self._process_frame(
                        mask_np, confidence, chunk_start, state, 0,
                        is_init=(ci == 0), frame_shape=mask_np.shape
                    )
                    predictions.append(pred)
                    occlusion_scores.append(occ)

                    last_masks = masks

                    for local_idx, _, masks in tqdm(
                        self.predictor.propagate_in_video(state),
                        desc=f"Chunk {ci+1}/{len(chunks)}",
                        leave=False
                    ):
                        if local_idx == 0:
                            continue

                        global_idx = chunk_start + local_idx
                        mask_np = (masks[0] > 0.0).cpu().numpy().squeeze()
                        confidence = float(torch.sigmoid(masks[0].max()).item())

                        pred, occ = self._process_frame(
                            mask_np, confidence, global_idx, state, local_idx,
                            is_init=False, frame_shape=mask_np.shape
                        )
                        predictions.append(pred)
                        occlusion_scores.append(occ)
                        last_masks = masks

                    carry_mask = (last_masks[0] > 0.0).cpu().numpy().squeeze()
                    self.predictor.reset_state(state)

            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        return predictions, occlusion_scores

    def _process_frame(self, mask: np.ndarray, confidence: float,
                       frame_idx: int, state, local_idx: int,
                       is_init: bool, frame_shape: tuple) -> Tuple[Optional[List[int]], float]:
        """
        Process frame with proper occlusion handling.
        Returns None when object is not visible (occluded/lost).
        """
        frame_h, frame_w = frame_shape[:2] if len(frame_shape) >= 2 else (480, 640)
        frame_area = frame_h * frame_w

        sam_bbox = self._mask_to_bbox_center(mask)
        bbox_area = sam_bbox[2] * sam_bbox[3]

        # Detect invalid detections
        is_empty = (sam_bbox[2] <= 0 or sam_bbox[3] <= 0)
        is_hallucinating = (bbox_area > 0.4 * frame_area)  # Covers >40% of frame
        is_too_small = (bbox_area < 100)  # Too small to be real

        # Validate against memory bank
        is_valid, boosted_conf = self.memory_bank.validate_detection(sam_bbox, confidence)

        if is_empty or is_hallucinating or is_too_small or not is_valid:
            confidence = 0.0

        # Update tracking state
        prev_state = self.tracking_state
        self._update_state(confidence)

        occlusion_score = 1.0 - confidence

        # === FIRST FRAME ===
        if is_init:
            self.kalman.update(sam_bbox)
            self.last_valid_bbox = sam_bbox.copy()
            self.memory_bank.set_initial(sam_bbox, mask)
            self.memory_bank.add_template(frame_idx, sam_bbox, mask, confidence)
            return self._to_xywh(sam_bbox), occlusion_score

        # === STATE-BASED PROCESSING ===

        if self.tracking_state == TrackingState.VISIBLE:
            # High confidence - trust SAM2
            self.kalman.update(sam_bbox)
            self.last_valid_bbox = sam_bbox.copy()
            self.memory_bank.add_template(frame_idx, sam_bbox, mask, confidence)
            self.frames_since_visible = 0
            return self._to_xywh(sam_bbox), occlusion_score

        elif self.tracking_state == TrackingState.UNCERTAIN:
            # Medium confidence - blend with Kalman
            kalman_pred = self.kalman.predict()
            alpha = confidence / self.confidence_threshold
            blended = alpha * sam_bbox + (1 - alpha) * kalman_pred
            self.kalman.update(blended)
            self.last_valid_bbox = blended.copy()
            self.frames_since_visible += 1
            return self._to_xywh(blended), occlusion_score

        elif self.tracking_state == TrackingState.OCCLUDED:
            # Object occluded - NO PREDICTION, just track with Kalman internally
            self.kalman.predict()  # Keep Kalman running
            self.frames_since_visible += 1
            return None, 1.0  # Return None = no detection

        elif self.tracking_state == TrackingState.LOST:
            # Lost - try re-detection with initial bbox
            self.frames_since_visible += 1

            if prev_state != TrackingState.LOST:
                print(f"\n  [Frame {frame_idx}] LOST - Attempting re-detection...")

            try:
                # Re-prompt with initial bbox
                _, _, new_masks = self.predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=local_idx, obj_id=1,
                    box=self.init_bbox_xyxy
                )
                new_mask = (new_masks[0] > 0.0).cpu().numpy().squeeze()
                new_conf = float(torch.sigmoid(new_masks[0].max()).item())
                new_bbox = self._mask_to_bbox_center(new_mask)

                # Validate re-detection with memory bank
                is_valid_redetect, boosted = self.memory_bank.validate_detection(new_bbox, new_conf)
                new_area = new_bbox[2] * new_bbox[3]

                if is_valid_redetect and new_conf > self.occlusion_threshold and new_area < 0.4 * frame_area:
                    print(f"  [Frame {frame_idx}] Re-detection SUCCESS (conf={new_conf:.2f} -> {boosted:.2f})")
                    self.kalman.initialize(new_bbox)
                    self.last_valid_bbox = new_bbox.copy()
                    self.tracking_state = TrackingState.UNCERTAIN
                    self.low_conf_count = 0
                    self.frames_since_visible = 0
                    return self._to_xywh(new_bbox), 1.0 - boosted
                else:
                    return None, 1.0  # Re-detection failed

            except Exception as e:
                return None, 1.0  # Return None on error

        return None, 1.0

    def _update_state(self, confidence: float):
        """Update tracking state machine."""
        if confidence >= self.confidence_threshold:
            self.tracking_state = TrackingState.VISIBLE
            self.low_conf_count = 0

        elif confidence >= self.occlusion_threshold:
            self.low_conf_count += 1
            if self.low_conf_count >= self.recovery_frames:
                self.tracking_state = TrackingState.UNCERTAIN

        elif confidence >= self.lost_threshold:
            self.low_conf_count += 1
            if self.low_conf_count >= self.recovery_frames:
                self.tracking_state = TrackingState.OCCLUDED

        else:
            self.low_conf_count += 1
            if self.low_conf_count >= self.recovery_frames * 2:
                self.tracking_state = TrackingState.LOST

    @staticmethod
    def _mask_to_bbox_center(mask: np.ndarray) -> np.ndarray:
        if mask.sum() == 0:
            return np.array([0, 0, 0, 0], dtype=np.float32)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return np.array([(x_min + x_max) / 2, (y_min + y_max) / 2,
                        x_max - x_min, y_max - y_min], dtype=np.float32)

    @staticmethod
    def _to_xywh(bbox: np.ndarray) -> List[int]:
        cx, cy, w, h = bbox
        return [int(cx - w/2), int(cy - h/2), int(w), int(h)]

    def _reset(self):
        self.kalman.reset()
        self.memory_bank.clear()
        self.tracking_state = TrackingState.VISIBLE
        self.low_conf_count = 0
        self.last_valid_bbox = None
        self.init_bbox_xyxy = None
        self.frames_since_visible = 0
