# Deepfake Detection - Model Training

This directory contains code and resources for training the deepfake detection model.

## System Requirements

- Python 3.10
- NVIDIA GPU (RTX 3060)
- CUDA and cuDNN (compatible with PyTorch/TensorFlow)

## Setup Instructions

### 1. Create Virtual Environment

```bash
# From the model-train directory
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On Linux/Mac
# source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Ensure pip is up to date
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Install PyTorch with CUDA support (if the requirements.txt version doesn't include it)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Dataset Preparation

Place your dataset in the `data` directory following this structure:

```
data/
├── real/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── fake/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── metadata.csv (optional)
```

### 4. Running the Training

```bash
# To train the model with default parameters
python train.py

# To train with custom parameters
python train.py --batch_size 32 --epochs 50 --model_type efficientnet
```

### 5. Exporting the Model

```bash
# Export the trained model for use in the FastAPI backend
python export_model.py --model_path ./models/best_model.pth --export_format onnx
```

## Project Structure

```
model-train/
├── data/                    # Dataset directory (to be populated)
├── models/                  # Saved model checkpoints
├── notebooks/               # Jupyter notebooks for exploration and analysis
├── src/                     # Source code
│   ├── data/                # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py       # Dataset classes
│   │   ├── augmentation.py  # Data augmentation strategies
│   │   └── utils.py         # Utility functions for data handling
│   ├── models/              # Model architectures
│   │   ├── __init__.py
│   │   ├── cnn.py           # CNN-based models
│   │   └── efficientnet.py  # EfficientNet-based models
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   ├── metrics.py       # Evaluation metrics
│   │   └── visualization.py # Visualization utilities
│   └── __init__.py
├── train.py                 # Main training script
├── evaluate.py              # Evaluation script
├── export_model.py          # Model export script
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Notes for Training

- The default configuration is optimized for RTX 3060 with 32GB system RAM
- Training logs will be saved to the `logs` directory
- Model checkpoints will be saved to the `models` directory
- Use TensorBoard to monitor training progress:
  ```
  tensorboard --logdir=logs
  ```
