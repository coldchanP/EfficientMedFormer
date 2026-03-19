# EfficientMedFormer Inference

This project provides a simple image inference script and a CPU benchmarking script for an EfficientMedFormer model.

## Files

- `model.py`: EfficientMedFormer model definition
- `inference.py`: Single-image inference script
- `benchmark_cpu_inference.py`: CPU inference benchmark
- `best_model.pth`: Trained checkpoint

## Installation

```bash
pip install -r requirements.txt
```

## Run Inference

```bash
python inference.py --image path/to/image.jpg --checkpoint best_model.pth --image-size 224
```

The script prints the predicted class index and the probability for each class.

## Run CPU Benchmark

```bash
python benchmark_cpu_inference.py --checkpoint best_model.pth --image-size 224
```
