"""
Configuration file for Motion-Aware SAM 2 project
"""
import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "datasets"
RESULTS_ROOT = PROJECT_ROOT / "results"
CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"

# Dataset paths
GOT10K_ROOT = DATA_ROOT / "GOT-10k"
LASOT_ROOT = DATA_ROOT / "LaSOT"
DAVIS_ROOT = DATA_ROOT / "DAVIS"

# =============================================================================
# SAM 2 CONFIGURATION
# =============================================================================
SAM2_CONFIG = {
    "model_type": "sam2.1_hiera_large",  # Options: sam2.1_hiera_tiny, sam2.1_hiera_small, sam2.1_hiera_base_plus, sam2.1_hiera_large
    "checkpoint": None,  # Will be downloaded automatically
    "device": "cuda",
}

# =============================================================================
# KALMAN FILTER CONFIGURATION
# =============================================================================
KALMAN_CONFIG = {
    # State: [x, y, w, h, vx, vy, vw, vh]
    "dim_x": 8,  # State dimension
    "dim_z": 4,  # Measurement dimension (x, y, w, h)

    # Process noise (how much we trust the motion model)
    "process_noise_position": 1.0,
    "process_noise_velocity": 0.1,

    # Measurement noise (how much we trust SAM 2's output)
    "measurement_noise": 1.0,

    # Score weighting
    "alpha_motion": 0.15,  # Weight for motion score
    "alpha_appearance": 0.85,  # Weight for appearance score (1 - alpha_motion)

    # Thresholds for quality-gated memory
    "tau_mask_iou": 0.5,  # Minimum mask IoU confidence
    "tau_motion": 0.7,  # Minimum motion score (IoU with predicted box)
    "tau_occlusion": 0.5,  # Maximum occlusion score (lower = more visible)
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
EVAL_CONFIG = {
    # GOT-10k metrics
    "got10k_iou_thresholds": [0.5, 0.75],  # For SR_0.5 and SR_0.75

    # LaSOT metrics
    "lasot_n_bins": 100,  # For AUC computation

    # General
    "save_visualizations": True,
    "visualization_freq": 50,  # Save every N frames
    "max_videos_per_dataset": None,  # None = all videos, or set a number for testing
}

# =============================================================================
# DATASET URLS
# =============================================================================
DATASET_URLS = {
    "got10k_test": "http://got-10k.aitestunion.com/downloads/GOT-10k_Test_000.zip",
    "got10k_val": "http://got-10k.aitestunion.com/downloads/GOT-10k_Val_000.zip",
    "lasot": "https://drive.google.com/drive/folders/1U9yvJNR5aWQU-sGRz2y1AqZA7pHKQ5jR",
}

# Create directories if they don't exist
for path in [DATA_ROOT, RESULTS_ROOT, CHECKPOINTS_ROOT, GOT10K_ROOT, LASOT_ROOT]:
    path.mkdir(parents=True, exist_ok=True)