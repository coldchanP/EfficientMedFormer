#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import EfficientMedFormer


class Config:
    """Placeholder for checkpoints that pickle the training Config object."""


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint
    raise TypeError("Unsupported checkpoint format.")


def sanitize_state_dict(state_dict):
    filtered = {}
    removed_keys = []
    for key, value in state_dict.items():
        if key.endswith("total_ops") or key.endswith("total_params"):
            removed_keys.append(key)
            continue
        filtered[key] = value
    return filtered, removed_keys


def infer_num_classes(state_dict):
    for key in ("classifier.0.weight", "classifier.weight"):
        weight = state_dict.get(key)
        if weight is not None:
            return int(weight.shape[0])
    raise KeyError("Could not infer num_classes from checkpoint.")


def load_model(checkpoint_path, image_size, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict, removed_keys = sanitize_state_dict(extract_state_dict(checkpoint))
    num_classes = infer_num_classes(state_dict)

    model = EfficientMedFormer(image_size=image_size, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, num_classes, removed_keys


def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_image(image_path, image_size, device):
    image = Image.open(image_path).convert("RGB")
    tensor = build_transform(image_size)(image).unsqueeze(0)
    return tensor.to(device)


def predict(model, image_tensor):
    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())
    return predicted_index, probabilities.cpu()


def parse_args():
    parser = argparse.ArgumentParser(description="Run image inference with EfficientMedFormer.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument(
        "--checkpoint",
        default="./best_model.pth",
        help="Path to checkpoint file.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square input image size.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use. Example: cpu, cuda, cuda:0",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    model, num_classes, removed_keys = load_model(
        checkpoint_path=checkpoint_path,
        image_size=args.image_size,
        device=device,
    )
    image_tensor = load_image(image_path, args.image_size, device)
    predicted_index, probabilities = predict(model, image_tensor)

    print(f"image: {image_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {device}")
    print(f"input_shape: {tuple(image_tensor.shape)}")
    print(f"num_classes: {num_classes}")
    print(f"predicted_class: {predicted_index}")
    print(f"removed_profiler_keys: {len(removed_keys)}")
    print("class_probabilities:")
    for class_index, probability in enumerate(probabilities.tolist()):
        print(f"  class_{class_index}: {probability:.6f}")


if __name__ == "__main__":
    main()
