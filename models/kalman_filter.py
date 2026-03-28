"""
Kalman Filter for Motion-Aware Object Tracking

State Vector: [x, y, w, h, vx, vy, vw, vh]
    - (x, y): center coordinates of bounding box
    - (w, h): width and height of bounding box
    - (vx, vy): velocity of center
    - (vw, vh): rate of change of width and height

Motion Model: Constant Velocity
    x_new = x + vx
    y_new = y + vy
    w_new = w + vw
    h_new = h + vh
"""

import numpy as np
from filterpy.kalman import KalmanFilter


class MotionAwareKalmanFilter:
    """
    Kalman Filter for tracking object bounding boxes with constant velocity model.

    This is the core component that makes SAM 2 motion-aware.
    """

    def __init__(self, config: dict):
        """
        Initialize the Kalman Filter.

        Args:
            config: Dictionary containing Kalman filter configuration
        """
        self.config = config
        self.kf = KalmanFilter(dim_x=config["dim_x"], dim_z=config["dim_z"])

        # State transition matrix (constant velocity model)
        # State: [x, y, w, h, vx, vy, vw, vh]
        # x_new = x + vx, y_new = y + vy, etc.
        dt = 1  # Time step (1 frame)
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],   # x = x + vx*dt
            [0, 1, 0, 0, 0,  dt, 0,  0],   # y = y + vy*dt
            [0, 0, 1, 0, 0,  0,  dt, 0],   # w = w + vw*dt
            [0, 0, 0, 1, 0,  0,  0,  dt],  # h = h + vh*dt
            [0, 0, 0, 0, 1,  0,  0,  0],   # vx = vx (constant)
            [0, 0, 0, 0, 0,  1,  0,  0],   # vy = vy (constant)
            [0, 0, 0, 0, 0,  0,  1,  0],   # vw = vw (constant)
            [0, 0, 0, 0, 0,  0,  0,  1],   # vh = vh (constant)
        ], dtype=np.float32)

        # Measurement matrix (we only observe x, y, w, h)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)

        # Process noise covariance matrix
        q_pos = config["process_noise_position"]
        q_vel = config["process_noise_velocity"]
        self.kf.Q = np.diag([q_pos, q_pos, q_pos, q_pos,
                            q_vel, q_vel, q_vel, q_vel]).astype(np.float32)

        # Measurement noise covariance matrix
        r = config["measurement_noise"]
        self.kf.R = np.diag([r, r, r, r]).astype(np.float32)

        # Initial covariance matrix (high uncertainty)
        self.kf.P *= 1000.0

        self.initialized = False
        self.alpha_motion = config["alpha_motion"]
        self.tau_motion = config["tau_motion"]

    def initialize(self, bbox: np.ndarray):
        """
        Initialize the Kalman filter with the first bounding box.

        Args:
            bbox: [x, y, w, h] - center x, center y, width, height
        """
        # State: [x, y, w, h, vx, vy, vw, vh]
        self.kf.x = np.array([
            bbox[0],  # x
            bbox[1],  # y
            bbox[2],  # w
            bbox[3],  # h
            0,        # vx (unknown, start at 0)
            0,        # vy
            0,        # vw
            0,        # vh
        ], dtype=np.float32).reshape(-1, 1)

        self.initialized = True

    def predict(self) -> np.ndarray:
        """
        Predict the next state (bounding box position).

        Returns:
            predicted_bbox: [x, y, w, h]
        """
        if not self.initialized:
            raise RuntimeError("Kalman filter not initialized. Call initialize() first.")

        self.kf.predict()
        return self.kf.x[:4].flatten()

    def update(self, bbox: np.ndarray):
        """
        Update the Kalman filter with a new measurement.

        Args:
            bbox: [x, y, w, h] - measured bounding box
        """
        if not self.initialized:
            raise RuntimeError("Kalman filter not initialized. Call initialize() first.")

        measurement = np.array(bbox, dtype=np.float32).reshape(-1, 1)
        self.kf.update(measurement)

    def get_state(self) -> np.ndarray:
        """
        Get the current state estimate.

        Returns:
            state: [x, y, w, h, vx, vy, vw, vh]
        """
        return self.kf.x.flatten()

    def get_velocity(self) -> np.ndarray:
        """
        Get the current velocity estimate.

        Returns:
            velocity: [vx, vy, vw, vh]
        """
        return self.kf.x[4:].flatten()

    def compute_motion_score(self, candidate_bbox: np.ndarray) -> float:
        """
        Compute motion consistency score between predicted box and candidate box.

        Args:
            candidate_bbox: [x, y, w, h] - candidate bounding box from SAM 2

        Returns:
            motion_score: IoU between predicted and candidate box (0 to 1)
        """
        predicted_bbox = self.kf.x[:4].flatten()
        return self._compute_iou(predicted_bbox, candidate_bbox)

    def score_candidates(self, candidates: list, iou_scores: list) -> tuple:
        """
        Score multiple mask candidates using motion + appearance.

        Args:
            candidates: List of bounding boxes [x, y, w, h] from mask candidates
            iou_scores: List of IoU confidence scores from SAM 2

        Returns:
            best_idx: Index of the best candidate
            scores: List of (final_score, motion_score, appearance_score) for each candidate
        """
        if len(candidates) == 0:
            return None, []

        scores = []
        for i, (bbox, iou_score) in enumerate(zip(candidates, iou_scores)):
            motion_score = self.compute_motion_score(bbox)

            # Combined score: alpha * motion + (1-alpha) * appearance
            final_score = (self.alpha_motion * motion_score +
                          (1 - self.alpha_motion) * iou_score)

            scores.append({
                "final_score": final_score,
                "motion_score": motion_score,
                "appearance_score": iou_score,
            })

        # Find best candidate
        best_idx = max(range(len(scores)), key=lambda i: scores[i]["final_score"])

        return best_idx, scores

    def should_store_in_memory(self, motion_score: float, iou_score: float,
                               occlusion_score: float) -> bool:
        """
        Decide whether to store this frame in SAM 2's memory bank.

        Quality-gated memory: only store if all scores pass thresholds.

        Args:
            motion_score: IoU between predicted and actual box
            iou_score: SAM 2's mask confidence
            occlusion_score: SAM 2's occlusion prediction (0 = visible, 1 = occluded)

        Returns:
            should_store: True if frame should be stored in memory
        """
        tau_mask = self.config["tau_mask_iou"]
        tau_motion = self.config["tau_motion"]
        tau_occlusion = self.config["tau_occlusion"]

        # All conditions must pass
        return (motion_score >= tau_motion and
                iou_score >= tau_mask and
                occlusion_score <= tau_occlusion)

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute IoU between two boxes in [x_center, y_center, w, h] format.

        Args:
            box1: [x, y, w, h]
            box2: [x, y, w, h]

        Returns:
            iou: Intersection over Union (0 to 1)
        """
        # Convert to [x1, y1, x2, y2] format
        box1_x1 = box1[0] - box1[2] / 2
        box1_y1 = box1[1] - box1[3] / 2
        box1_x2 = box1[0] + box1[2] / 2
        box1_y2 = box1[1] + box1[3] / 2

        box2_x1 = box2[0] - box2[2] / 2
        box2_y1 = box2[1] - box2[3] / 2
        box2_x2 = box2[0] + box2[2] / 2
        box2_y2 = box2[1] + box2[3] / 2

        # Intersection
        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Union
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def reset(self):
        """Reset the Kalman filter to uninitialized state."""
        self.initialized = False
        self.kf.x = np.zeros((8, 1), dtype=np.float32)
        self.kf.P = np.eye(8, dtype=np.float32) * 1000.0


def mask_to_bbox(mask: np.ndarray) -> np.ndarray:
    """
    Convert a binary mask to bounding box [x_center, y_center, w, h].

    Args:
        mask: Binary mask (H, W) with 1s for object pixels

    Returns:
        bbox: [x_center, y_center, width, height]
    """
    if mask.sum() == 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)

    # Find non-zero pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Convert to center format
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return np.array([x_center, y_center, width, height], dtype=np.float32)


def bbox_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """
    Convert bbox from [x_center, y_center, w, h] to [x1, y1, x2, y2].

    Args:
        bbox: [x_center, y_center, width, height]

    Returns:
        bbox_xyxy: [x1, y1, x2, y2]
    """
    x, y, w, h = bbox
    return np.array([x - w/2, y - h/2, x + w/2, y + h/2], dtype=np.float32)


def xyxy_to_bbox(bbox_xyxy: np.ndarray) -> np.ndarray:
    """
    Convert bbox from [x1, y1, x2, y2] to [x_center, y_center, w, h].

    Args:
        bbox_xyxy: [x1, y1, x2, y2]

    Returns:
        bbox: [x_center, y_center, width, height]
    """
    x1, y1, x2, y2 = bbox_xyxy
    return np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dtype=np.float32)
