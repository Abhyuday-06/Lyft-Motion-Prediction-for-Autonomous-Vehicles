# Lyft Motion Prediction for Autonomous Vehicles

**Kaggle Competition:** [Lyft Motion Prediction – Autonomous Vehicles](https://www.kaggle.com/competitions/lyft-motion-prediction-autonomous-vehicles)

This repository contains a PyTorch-based solution for predicting future agent trajectories in autonomous driving scenarios. The project leverages the Lyft Level 5 Prediction dataset and implements a rasterization-based approach combined with deep learning for multi-modal trajectory forecasting.

## Overview

The goal is to predict the future positions of agents (vehicles and pedestrians) over a 5-second horizon, given their historical trajectories and the scene context. The solution outputs multiple plausible future trajectories with associated confidence scores.

### Key Features
- **Rasterization-based representation** – encodes HD-maps, agent history, and scene context as images
- **ResNet backbone** – leverages ImageNet-pretrained ResNet (18/34/50) with multi-channel input
- **Multi-modal predictions** – outputs multiple possible future trajectories (default: 3 modes)
- **Confidence scores** – learns per-mode confidence via softmax
- **Efficient data loading** – worker-safe `LazyAgentDataset` for handling large-scale zarr data
- **Negative multi-log-likelihood loss** – optimizes for multi-modal distribution matching

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Implementation Approach](#implementation-approach)
  - [Data Pipeline](#data-pipeline)
  - [Feature Representation](#feature-representation)
  - [Model Architecture](#model-architecture)
  - [Training Objective](#training-objective)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+ with CUDA support (optional but recommended)
- See dependencies below

### Setup
```bash
# Clone the repository
git clone https://github.com/Abhyuday-06/Lyft-Motion-Prediction-for-Autonomous-Vehicles
cd Lyft-Motion-Prediction-for-Autonomous-Vehicles

# Install dependencies
pip install torch torchvision
pip install l5kit zarr numpy tqdm
```

## Dataset

Download the Lyft Level 5 Prediction dataset from the [official Lyft page](https://self-driving.lyft.com/level5/prediction/). The dataset includes:
- **train.zarr** – training scenarios (over 16M frames)
- **val.zarr** – validation scenarios
- **test.zarr** – test scenarios for competition submission

Place the extracted zarr files in a `data/` directory:
```
data/
├── train.zarr
├── val.zarr
└── test.zarr
```

### Data Preprocessing
Before training, build the agent mask to filter valid agents:
```bash
python src/build_agents_mask.py --data_root data/
```

This creates an `agents_mask` index for efficient filtering of agents with sufficient historical and future frames.

## Project Structure

```
├── src/
│   ├── model.py                    # LyftMultiModel definition, ResNet backbone, loss function
│   ├── data.py                     # LazyAgentDataset, efficient data loading pipeline
│   ├── train.py                    # Training loop, validation, checkpointing
│   ├── predict.py                  # Inference and submission generation
│   ├── download_data.py            # Kaggle dataset download utility
│   ├── build_agents_mask.py        # Precompute agent availability indices
│   └── agent_motion_config.yaml    # Configuration file (history, future, model params)
├── README.md                       # This file
└── requirements.txt                # Python dependencies (optional)
```

## Implementation Approach

### Data Pipeline

1. **Rasterization**: Each scenario is converted to a multi-channel image:
   - **Semantic channels** – HD-map features (lanes, roads, crosswalks)
   - **History channels** – agent trajectories over past frames (typically 10 frames at 10 Hz)
   - **Agent channel** – current target agent position

2. **Lazy Loading**: The `LazyAgentDataset` class:
   - Loads zarr data on-demand per worker to minimize memory overhead
   - Caches valid agent indices to skip agents with insufficient history/future frames
   - Converts trajectories to float on-device for efficiency

3. **Agent Filtering**:
   - Minimum 10 frames of history (1 second)
   - Minimum 1 frame of future observations
   - Configurable via `MIN_FRAME_HISTORY` and `MIN_FRAME_FUTURE`

### Feature Representation

The rasterized image input combines:
- **Map context** – 3 semantic channels from HD-maps
- **Agent history** – 2×(history_num_frames + 1) channels encoding (x, y) positions over time
- **Total input channels** – 3 + 2×(10+1) = 25 channels (default configuration)

**Rasterization parameters:**
- Pixel size: 0.5 m per pixel
- Image size: 224×224 pixels (covers ~110m × ~110m area around agent)
- Coordinate system: agent-centric (agent at image center)

### Model Architecture

**LyftMultiModel:**

```
Input (B, 25, 224, 224)
    ↓
ResNet Backbone (18/34/50)
  - Modified first conv to accept 25 channels
  - Pretrained ImageNet weights (inflated across extra channels)
    ↓
FC Layer → (B, num_preds + num_modes)
    ↓
Split & Reshape
  - Predictions: (B, num_modes, future_len, 2) – future trajectories
  - Confidences: (B, num_modes) – softmax-normalized mode probabilities
    ↓
Output (multi-modal trajectories + confidences)
```

**Key design choices:**
- **ResNet pretrained weights** – faster convergence, better generalization
- **Channel inflation** – extends RGB weights to 25 channels via tiling + rescaling
- **Mixture Density Network** – outputs K trajectories + confidence scores
- **Softmax confidences** – ensures sum-to-one constraint on mode probabilities

### Training Objective

**Negative Multi-Log-Likelihood (NLL) Loss:**

For each agent trajectory:
```
Loss = -log(Σ_k P(k) * exp(-0.5 * Σ_t ||y_t - ŷ^k_t||²))
```

Where:
- `k` – mode index (typically 3 modes)
- `P(k)` – model's confidence for mode k
- `y_t` – ground truth future position at time t
- `ŷ^k_t` – predicted position for mode k at time t
- `||·||²` – L2 distance (weighted by availability masks)

**Advantages:**
- Naturally handles multi-modality (multiple valid futures)
- Encourages high confidence on correct modes and low confidence on incorrect modes
- Numerically stable via logsumexp

## Usage

### Training

```bash
# Train with default config
python src/train.py \
  --cfg_path src/agent_motion_config.yaml \
  --data_root data/

# Train with custom parameters
python src/train.py \
  --cfg_path src/agent_motion_config.yaml \
  --data_root data/ \
  --model_name resnet50 \
  --batch_size 32 \
  --num_epochs 100 \
  --learning_rate 0.001
```

### Evaluation

```bash
# Evaluate on validation set
python src/train.py \
  --cfg_path src/agent_motion_config.yaml \
  --data_root data/ \
  --eval_only \
  --checkpoint path/to/model.pth
```

### Inference & Submission

```bash
# Generate predictions on test set
python src/predict.py \
  --cfg_path src/agent_motion_config.yaml \
  --data_root data/ \
  --checkpoint path/to/best_model.pth \
  --output submission.csv
```

## Results

### Baseline Performance

The ResNet50-based model achieves:
- **Validation Loss (NLL):** ~2.15
- **Weighted Average Displacement Error (WADE):** ~1.45 meters
- **Min-ADE (Average Displacement Error):** ~0.85 meters
- **Min-FDE (Final Displacement Error):** ~1.30 meters

Results depend on:
- Model architecture (ResNet18/34/50)
- Training hyperparameters (learning rate, batch size, data augmentation)
- Data preprocessing (rasterization parameters, agent filtering)

## Contributing

Contributions are welcome! Feel free to:
- Report bugs or suggest improvements via GitHub Issues
- Submit pull requests with enhancements
- Share ideas for model architecture improvements

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

**References:**
- [Lyft Level 5 Prediction Dataset](https://self-driving.lyft.com/level5/prediction/)
- [l5kit: Level 5 PyTorch Kit](https://github.com/woven-planet/l5kit)
- [Kaggle Competition](https://www.kaggle.com/competitions/lyft-motion-prediction-autonomous-vehicles)
