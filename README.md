# EfficientMedFormer

This project includes model definition, training code, and a simple inference script for the `pneumoniamnist` task.

## Files

- `model.py`: EfficientMedFormer model definition
- `train_pneumoniamnist_efficientmedformer.py`: training script
- `inference.py`: single-image inference script

## Installation

```bash
pip install -r requirements.txt
```

## Training

Run training with:

```bash
python train_pneumoniamnist_efficientmedformer.py
```

The main training settings are defined in the `Config` class inside [train_pneumoniamnist_efficientmedformer.py](C:/Users/tjdck/Downloads/zd/train_pneumoniamnist_efficientmedformer.py).

Important fields:

- `data_file`: path to the dataset file
- `save_dir`: folder where checkpoints are saved
- `log_dir`: folder where training logs are saved
- `image_size`: input image size
- `batch_size`: training batch size
- `epochs`: number of training epochs
- `device`: training device such as `cpu` or `cuda:0`

## How To Set Training Folders

Edit these values in the `Config` class:

```python
data_file = '../datasets/Medmnist/pneumoniamnist_224.npz'
save_dir = './checkpoints_pneumoniamnist_edgenext'
log_dir = './logs_pneumoniamnist_edgenext'
```

Example:

```python
data_file = 'D:/datasets/pneumoniamnist_224.npz'
save_dir = './outputs/checkpoints'
log_dir = './outputs/logs'
```

## Inference

Run inference with:

```bash
python inference.py --image path/to/image.jpg --checkpoint best_model.pth --image-size 224
```

The script prints the predicted class index and the probability for each class.
