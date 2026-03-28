"""
SAM 2 Video Tracker Wrapper

This module wraps SAM 2 for video object tracking with optional
Kalman filter integration for motion-aware tracking.
"""

import numpy as np
import torch
from typing import Optional, List, Tuple, Dict
from pathlib import Path

try:
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM 2 not installed. Install with: pip install segment-anything-2")

from .kalman_filter import MotionAwareKalmanFilter, mask_to_bbox, xyxy_to_bbox


class SAM2VideoTracker:
    """
    SAM 2 Video Tracker with optional Kalman filter integration.

    Modes:
    - baseline: Standard SAM 2 (no motion awareness)
    - kalman: SAM 2 + Kalman filter for motion-aware tracking
    """

    def __init__(
        self,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint: str = None,
        device: str = "cuda",
        mode: str = "baseline",  # "baseline" or "kalman"
        kalman_config: dict = None,
    ):
        """
        Initialize the SAM 2 Video Tracker.

        Args:
            model_cfg: Path to SAM 2 model config
            checkpoint: Path to SAM 2 checkpoint
            device: Device to run on ("cuda" or "cpu")
            mode: "baseline" or "kalman"
            kalman_config: Configuration dict for Kalman filter
        """
        self.device = device
        self.mode = mode

        # Initialize SAM 2
        if SAM2_AVAILABLE:
            self.predictor = build_sam2_video_predictor(
                model_cfg,
                checkpoint,
                device=device,
            )
        else:
            self.predictor = None
            print("SAM 2 not available - using dummy mode for testing")

        # Initialize Kalman filter if in kalman mode
        self.kalman_filter = None
        if mode == "kalman" and kalman_config is not None:
            self.kalman_filter = MotionAwareKalmanFilter(kalman_config)

        # Tracking state
        self.video_state = None
        self.frame_idx = 0
        self.tracking_active = False

    def initialize_video(self, video_path: str = None, frames: np.ndarray = None):
        """
        Initialize tracking on a new video.

        Args:
            video_path: Path to video file
            frames: Alternatively, numpy array of frames (N, H, W, 3)
        """
        if self.predictor is None:
            return

        if video_path is not None:
            self.video_state = self.predictor.init_state(video_path)
        elif frames is not None:
            # For frame-by-frame processing
            self.video_state = self.predictor.init_state(frames)

        self.frame_idx = 0
        self.tracking_active = False

        if self.kalman_filter is not None:
            self.kalman_filter.reset()

    def initialize_tracking(
        self,
        frame_idx: int,
        point: Tuple[int, int] = None,
        bbox: np.ndarray = None,
        mask: np.ndarray = None,
        obj_id: int = 1,
    ) -> Tuple[np.ndarray, float]:
        """
        Initialize tracking with a prompt on a specific frame.

        Args:
            frame_idx: Frame index to initialize on
            point: (x, y) point prompt
            bbox: [x1, y1, x2, y2] bounding box prompt
            mask: Binary mask prompt
            obj_id: Object ID for multi-object tracking

        Returns:
            mask: Predicted segmentation mask
            iou_score: Confidence score
        """
        if self.predictor is None:
            # Return dummy output for testing
            return np.zeros((480, 640), dtype=np.uint8), 0.9

        # Prepare prompts
        points = None
        labels = None
        box = None

        if point is not None:
            points = np.array([[point]], dtype=np.float32)
            labels = np.array([[1]], dtype=np.int32)  # 1 = positive point

        if bbox is not None:
            box = np.array([bbox], dtype=np.float32)

        # Add object to tracking
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.video_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            box=box,
        )

        # Get mask from logits
        mask = (out_mask_logits[0] > 0).cpu().numpy().squeeze()
        iou_score = torch.sigmoid(out_mask_logits[0].max()).item()

        # Initialize Kalman filter with first bbox
        if self.kalman_filter is not None and mask.sum() > 0:
            bbox_xywh = mask_to_bbox(mask)
            self.kalman_filter.initialize(bbox_xywh)

        self.tracking_active = True
        self.frame_idx = frame_idx

        return mask, iou_score

    def track_frame(
        self,
        frame_idx: int,
        obj_id: int = 1,
    ) -> Dict:
        """
        Track the object to a new frame.

        Args:
            frame_idx: Target frame index
            obj_id: Object ID to track

        Returns:
            result: Dictionary containing:
                - mask: Segmentation mask
                - iou_score: Confidence score
                - occlusion_score: Occlusion prediction
                - motion_score: (if kalman mode) Motion consistency score
                - bbox: Bounding box [x, y, w, h]
                - should_store: Whether to store in memory
        """
        if self.predictor is None:
            # Dummy output for testing
            return {
                "mask": np.zeros((480, 640), dtype=np.uint8),
                "iou_score": 0.9,
                "occlusion_score": 0.1,
                "motion_score": 0.9 if self.mode == "kalman" else None,
                "bbox": np.array([320, 240, 100, 100]),
                "should_store": True,
            }

        result = {}

        # Propagate to frame
        out_frame_idx, out_obj_ids, out_mask_logits = self.predictor.propagate_in_video(
            inference_state=self.video_state,
            start_frame_idx=frame_idx,
            max_frame_num_to_track=1,
        )

        # Get mask and scores
        mask_logits = out_mask_logits[0]  # (1, H, W)
        mask = (mask_logits > 0).cpu().numpy().squeeze()
        iou_score = torch.sigmoid(mask_logits.max()).item()

        # Estimate occlusion score (use inverse of max logit confidence)
        occlusion_score = 1.0 - iou_score

        result["mask"] = mask
        result["iou_score"] = iou_score
        result["occlusion_score"] = occlusion_score

        # Get bounding box from mask
        if mask.sum() > 0:
            bbox = mask_to_bbox(mask)
        else:
            # No mask detected - use Kalman prediction if available
            if self.kalman_filter is not None and self.kalman_filter.initialized:
                bbox = self.kalman_filter.predict()
            else:
                bbox = np.array([0, 0, 0, 0])

        result["bbox"] = bbox

        # Kalman filter processing
        if self.mode == "kalman" and self.kalman_filter is not None:
            if self.kalman_filter.initialized:
                # Predict next position
                predicted_bbox = self.kalman_filter.predict()

                # Compute motion score
                motion_score = self.kalman_filter.compute_motion_score(bbox)
                result["motion_score"] = motion_score

                # Decide whether to store in memory
                should_store = self.kalman_filter.should_store_in_memory(
                    motion_score, iou_score, occlusion_score
                )
                result["should_store"] = should_store

                # Update Kalman filter with measurement
                self.kalman_filter.update(bbox)
            else:
                result["motion_score"] = 1.0
                result["should_store"] = True
        else:
            result["motion_score"] = None
            result["should_store"] = True  # Baseline always stores

        self.frame_idx = frame_idx

        return result

    def track_video(
        self,
        frames: List[np.ndarray],
        init_bbox: np.ndarray,
        obj_id: int = 1,
    ) -> List[Dict]:
        """
        Track object through entire video.

        Args:
            frames: List of frames (H, W, 3)
            init_bbox: Initial bounding box [x1, y1, x2, y2]
            obj_id: Object ID

        Returns:
            results: List of tracking results per frame
        """
        if len(frames) == 0:
            return []

        results = []

        # Initialize on first frame
        mask, iou_score = self.initialize_tracking(
            frame_idx=0,
            bbox=init_bbox,
            obj_id=obj_id,
        )

        results.append({
            "frame_idx": 0,
            "mask": mask,
            "iou_score": iou_score,
            "occlusion_score": 0.0,
            "motion_score": 1.0 if self.mode == "kalman" else None,
            "bbox": mask_to_bbox(mask) if mask.sum() > 0 else xyxy_to_bbox(init_bbox),
            "should_store": True,
        })

        # Track through remaining frames
        for frame_idx in range(1, len(frames)):
            result = self.track_frame(frame_idx, obj_id=obj_id)
            result["frame_idx"] = frame_idx
            results.append(result)

        return results

    def reset(self):
        """Reset the tracker state."""
        self.video_state = None
        self.frame_idx = 0
        self.tracking_active = False

        if self.kalman_filter is not None:
            self.kalman_filter.reset()


class SAM2TrackerSimple:
    """
    Simplified SAM 2 tracker for frame-by-frame processing.

    This version doesn't require video initialization and works
    directly with individual frames.
    """

    def __init__(
        self,
        model_type: str = "sam2.1_hiera_large",
        device: str = "cuda",
        mode: str = "baseline",
        kalman_config: dict = None,
    ):
        self.device = device
        self.mode = mode

        # Try to load SAM 2
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_model = build_sam2(model_type, device=device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            self.sam2_available = True
        except Exception as e:
            print(f"Warning: Could not load SAM 2: {e}")
            self.predictor = None
            self.sam2_available = False

        # Initialize Kalman filter
        self.kalman_filter = None
        if mode == "kalman" and kalman_config is not None:
            self.kalman_filter = MotionAwareKalmanFilter(kalman_config)

        self.prev_mask = None
        self.initialized = False

    def initialize(self, frame: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Initialize tracking on first frame.

        Args:
            frame: First frame (H, W, 3) RGB
            bbox: Initial bounding box [x1, y1, x2, y2]

        Returns:
            mask: Predicted mask
            iou_score: Confidence
        """
        if not self.sam2_available:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            return mask, 0.9

        self.predictor.set_image(frame)

        masks, iou_scores, _ = self.predictor.predict(
            box=bbox,
            multimask_output=True,
        )

        # Select best mask
        best_idx = np.argmax(iou_scores)
        mask = masks[best_idx]
        iou_score = iou_scores[best_idx]

        # Initialize Kalman filter
        if self.kalman_filter is not None:
            bbox_xywh = mask_to_bbox(mask)
            self.kalman_filter.initialize(bbox_xywh)

        self.prev_mask = mask
        self.initialized = True

        return mask, iou_score

    def track(self, frame: np.ndarray) -> Dict:
        """
        Track object in next frame.

        Args:
            frame: Current frame (H, W, 3) RGB

        Returns:
            result: Tracking result dictionary
        """
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")

        result = {}

        if not self.sam2_available:
            result["mask"] = np.zeros(frame.shape[:2], dtype=np.uint8)
            result["iou_score"] = 0.9
            result["bbox"] = np.array([0, 0, 0, 0])
            result["motion_score"] = 1.0 if self.mode == "kalman" else None
            return result

        self.predictor.set_image(frame)

        # Use previous mask as prompt
        masks, iou_scores, _ = self.predictor.predict(
            mask_input=self.prev_mask[np.newaxis, :, :],
            multimask_output=True,
        )

        # In Kalman mode, score candidates by motion
        if self.mode == "kalman" and self.kalman_filter is not None:
            # Get predicted position
            predicted_bbox = self.kalman_filter.predict()

            # Score each candidate
            candidate_bboxes = [mask_to_bbox(m) for m in masks]
            best_idx, scores = self.kalman_filter.score_candidates(
                candidate_bboxes, iou_scores.tolist()
            )

            mask = masks[best_idx]
            iou_score = iou_scores[best_idx]
            motion_score = scores[best_idx]["motion_score"]

            # Update Kalman
            self.kalman_filter.update(mask_to_bbox(mask))

            result["motion_score"] = motion_score
        else:
            # Baseline: just pick highest IoU
            best_idx = np.argmax(iou_scores)
            mask = masks[best_idx]
            iou_score = iou_scores[best_idx]
            result["motion_score"] = None

        result["mask"] = mask
        result["iou_score"] = float(iou_score)
        result["bbox"] = mask_to_bbox(mask)

        self.prev_mask = mask

        return result

    def reset(self):
        """Reset tracker state."""
        self.prev_mask = None
        self.initialized = False
        if self.kalman_filter is not None:
            self.kalman_filter.reset()
