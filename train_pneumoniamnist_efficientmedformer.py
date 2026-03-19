#!/usr/bin/env python3
"""
EdgeNext - {description}
- 데이터셋: {dataset_name_upper} (.npz 형식)
- 의료 이미지: 28x28 RGB 이미지 분류
- 모델: timm EdgeNext
"""
print('[Start] EdgeNext - {description} 시작!')

import os
import sys
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import math
import time
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

try:
    from thop import profile
except ImportError:
    profile = None

# Mixed Precision Training
from torch.cuda.amp import GradScaler
from torch.amp import autocast

# timm import
from model import EfficientMedFormer

# 상위 디렉토리의 모듈들 import
sys.path.append('..')

class Config:
    # 기본 설정
    seed = 42
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataset_name = 'pneumoniamnist'
    dataset_name_upper = 'PNEUMONIAMNIST'
    
    # 데이터 설정
    data_file = '../datasets/Medmnist/pneumoniamnist_224.npz'
    batch_size = 128
    effective_batch_size = 256
    num_workers = 8
    
    # 학습 설정
    epochs = 100
    base_lr = 0.001
    min_lr = 0.0001
    weight_decay = 0.01
    
    # Mixed Precision Training
    use_amp = True
    
    # Gradient Accumulation
    gradient_accumulation_steps = max(1, effective_batch_size // batch_size)
    
    # 정규화 기법
    label_smoothing = 0.05
    
    # 모델 설정
    image_size = 224
    
    # 저장 설정
    save_dir = './checkpoints_pneumoniamnist_edgenext'
    log_dir = './logs_pneumoniamnist_edgenext'
    
    # Warmup 설정
    warmup_epochs = 5
    warmup_start_lr = 0.0001
    
    # Gradient Clipping
    max_grad_norm = 1.0

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class TrainingLogger:
    def __init__(self, log_dir, dataset_name):
        self.log_dir = log_dir
        self.dataset_name = dataset_name
        self.logs = []
        self.start_time = time.time()

    def log_epoch(self, epoch, train_loss, train_metrics, val_loss, val_metrics, lr, epoch_time):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed_time = time.time() - self.start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, sec = divmod(rem, 60)
        log_entry = {
            'epoch': epoch + 1, 
            'train_loss': train_loss, 
            'train_oa': train_metrics['oa'],
            'train_auc': train_metrics['auc'],
            'train_precision': train_metrics['precision'],
            'train_sensitivity': train_metrics['sensitivity'],
            'train_specificity': train_metrics['specificity'],
            'train_f1': train_metrics['f1'],
            'val_loss': val_loss, 
            'val_oa': val_metrics['oa'],
            'val_auc': val_metrics['auc'],
            'val_precision': val_metrics['precision'],
            'val_sensitivity': val_metrics['sensitivity'],
            'val_specificity': val_metrics['specificity'],
            'val_f1': val_metrics['f1'],
            'learning_rate': lr,
            'epoch_time': epoch_time, 
            'timestamp': current_time,
            'elapsed_time': f"{int(hours)}h {int(minutes)}m {int(sec)}s"
        }
        self.logs.append(log_entry)

    def save_csv(self, filename=None):
        if filename is None:
            filename = f'training_log_edgenext_{self.dataset_name}.csv'
        if self.logs:
            df = pd.DataFrame(self.logs)
            filepath = os.path.join(self.log_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"[Log] {self.dataset_name} 학습 기록 저장: {filepath}")
            return filepath
        return None

class MedMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image, mode='RGB')
        elif len(image.shape) == 2:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image_rgb = np.stack([image, image, image], axis=2)
            image = Image.fromarray(image_rgb, mode='RGB')
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        if self.transform:
            image = self.transform(image)
        
        label = int(label)
        return image, label

def load_medmnist_data(config):
    print(f"[Data] {config.dataset_name_upper} 데이터 로딩 시작...")
    data_file = config.data_file
    if not os.path.exists(data_file):
        print(f"[Error] 파일을 찾을 수 없습니다: {data_file}")
        raise FileNotFoundError(f"{config.dataset_name} 파일이 없습니다: {data_file}")
    
    data = np.load(data_file)
    train_images_orig = data['train_images']
    train_labels_orig = data['train_labels'].flatten().astype(np.int64)
    val_images_orig = data['val_images']
    val_labels_orig = data['val_labels'].flatten().astype(np.int64)
    test_images_orig = data['test_images']
    test_labels_orig = data['test_labels'].flatten().astype(np.int64)
    
    # 기존 제공된 train / val / test를 모두 합친 뒤
    # train : val : test = 7 : 2 : 1 비율로 재분할
    all_images = np.concatenate([train_images_orig, val_images_orig, test_images_orig], axis=0)
    all_labels = np.concatenate([train_labels_orig, val_labels_orig, test_labels_orig], axis=0)
    
    # 1) 전체 데이터에서 10%를 test로 분리
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_images,
        all_labels,
        test_size=0.1,
        random_state=config.seed,
        stratify=all_labels
    )
    
    # 2) 남은 90%를 train : val = 7 : 2 비율로 분리 (val 비율 = 2/9)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=2.0 / 9.0,
        random_state=config.seed,
        stratify=y_temp
    )
    
    print(f"[Data Split] Total: {len(all_images)} | "
          f"Train: {len(X_train)} (70%) | Val: {len(X_val)} (20%) | Test: {len(X_test)} (10%)")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def get_transforms(config):
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, val_transforms

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size) for k in topk]
        return res

def calculate_metrics(all_probs, all_targets, num_classes):
    try:
        predictions = np.argmax(all_probs, axis=1)
        oa = (predictions == all_targets).mean() * 100.0
        
        try:
            from sklearn.metrics import roc_curve, auc as auc_func
            auc_scores = []
            for class_idx in range(num_classes):
                binary_targets = (all_targets == class_idx).astype(int)
                class_probs = all_probs[:, class_idx]
                if np.sum(binary_targets) > 0 and np.sum(binary_targets) < len(binary_targets):
                    fpr, tpr, _ = roc_curve(binary_targets, class_probs)
                    auc_scores.append(auc_func(fpr, tpr))
                else:
                    auc_scores.append(0.5)
            auc = np.mean(auc_scores) * 100.0
        except:
            auc = 0.0
        
        cm = confusion_matrix(all_targets, predictions, labels=range(num_classes))
        precisions, sensitivities, specificities, f1_scores = [], [], [], []
        
        for class_idx in range(num_classes):
            tp = cm[class_idx, class_idx]
            fp = cm[:, class_idx].sum() - tp
            fn = cm[class_idx, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1_class = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
            
            precisions.append(prec)
            sensitivities.append(sens)
            specificities.append(spec)
            f1_scores.append(f1_class)
        
        return {
            'oa': oa, 'auc': auc, 'precision': np.mean(precisions) * 100.0,
            'sensitivity': np.mean(sensitivities) * 100.0, 
            'specificity': np.mean(specificities) * 100.0, 
            'f1': np.mean(f1_scores) * 100.0
        }
    except:
        return {'oa': 0.0, 'auc': 0.0, 'precision': 0.0, 'sensitivity': 0.0, 'specificity': 0.0, 'f1': 0.0}

def train_epoch(model, train_loader, optimizer, scheduler, criterion, epoch, config, scaler, global_step):
    model.train()
    total_loss, total_acc1, total_samples = 0, 0, 0
    all_probs, all_targets = [], []
    train_bar = tqdm(train_loader, desc=f"Train {epoch + 1}/{config.epochs}")
    
    for batch_idx, (data, target) in enumerate(train_bar):
        data, target = data.to(config.device), target.to(config.device)
        
        with autocast('cuda', enabled=config.use_amp):
            output = model(data)
            loss = criterion(output, target)
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
        
        probs = F.softmax(output, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(target.detach().cpu().numpy())
        
        acc1 = accuracy(output, target, topk=(1,))[0]
        total_loss += loss.item() * config.gradient_accumulation_steps * data.size(0)
        total_acc1 += acc1.item() * data.size(0)
        total_samples += data.size(0)
        
        train_bar.set_postfix({
            'Loss': f'{loss.item()*config.gradient_accumulation_steps:.4f}',
            'Acc1': f'{acc1.item():.2f}%'
        })
        
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    train_metrics = calculate_metrics(all_probs, all_targets, config.num_classes)
    return total_loss / total_samples, total_acc1 / total_samples, train_metrics, global_step

def validate_epoch(model, val_loader, criterion, epoch, config):
    model.eval()
    total_loss, total_acc1, total_samples = 0, 0, 0
    all_probs, all_targets = [], []
    val_bar = tqdm(val_loader, desc=f"Val {epoch + 1}/{config.epochs}")

    with torch.no_grad():
        for data, target in val_bar:
            data, target = data.to(config.device), target.to(config.device)
            with autocast('cuda', enabled=config.use_amp):
                output = model(data)
                loss = criterion(output, target)
            
            probs = F.softmax(output, dim=1).detach().cpu().numpy()
            all_probs.append(probs)
            all_targets.append(target.detach().cpu().numpy())
            
            acc1 = accuracy(output, target, topk=(1,))[0]
            total_loss += loss.item() * data.size(0)
            total_acc1 += acc1.item() * data.size(0)
            total_samples += data.size(0)
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    val_metrics = calculate_metrics(all_probs, all_targets, config.num_classes)
    return total_loss/total_samples, total_acc1/total_samples, val_metrics

def save_checkpoint(model, epoch, val_acc1, config, is_best=False):
    if not is_best:
        return None
    os.makedirs(config.save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_acc1': val_acc1,
        'config': config
    }
    filepath = os.path.join(config.save_dir, f'best_model_edgenext_{config.dataset_name}.pth')
    torch.save(checkpoint, filepath)
    print(f"[Best] 최고 성능 모델 저장! (Acc1: {val_acc1:.2f}%) -> {filepath}")
    return filepath

def get_lr_scheduler(optimizer, config, total_iters, train_loader):
    iters_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    warmup_iters = config.warmup_epochs * iters_per_epoch
    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return config.warmup_start_lr / config.base_lr + (1 - config.warmup_start_lr / config.base_lr) * current_iter / warmup_iters
        progress = (current_iter - warmup_iters) / (total_iters - warmup_iters)
        return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - config.min_lr / config.base_lr) + config.min_lr / config.base_lr
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main():
    config = Config()
    seed_everything(config.seed)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_medmnist_data(config)
    num_classes = len(np.unique(train_labels))
    config.num_classes = num_classes

    train_transforms, val_transforms = get_transforms(config)
    train_dataset = MedMNISTDataset(train_images, train_labels, transform=train_transforms)
    val_dataset = MedMNISTDataset(val_images, val_labels, transform=val_transforms)
    test_dataset = MedMNISTDataset(test_images, test_labels, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, pin_memory=True)

    print(f"[Model] EfficientMedFormer 모델 로딩 중...")
    model = EfficientMedFormer(
        image_size=config.image_size,
        num_classes=num_classes,
        channels=3,
        kernel_size=5
    ).to(config.device)
    
    # ----- Model complexity summary (Params / FLOPs) -----
    dummy_input = torch.randn(1, 3, config.image_size, config.image_size).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters: {total_params / 1e6:.3f} M")
    if profile is not None:
        try:
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            print(f"[Model] FLOPs (per image @ {config.image_size}x{config.image_size}): "
                  f"{flops / 1e6:.3f} MFLOPs ({flops / 1e9:.3f} GFLOPs)")
        except Exception as e:
            print(f"[Model] FLOPs 계산 중 오류 발생: {e}")
    else:
        print("[Model] thop 패키지를 찾을 수 없어 FLOPs는 계산하지 않습니다.")
    
    print(f"[Model] EfficientMedFormer 모델 로드 완료!")

    optimizer = optim.AdamW(model.parameters(), lr=config.base_lr, weight_decay=config.weight_decay)
    total_iters = len(train_loader) // config.gradient_accumulation_steps * config.epochs
    scheduler = get_lr_scheduler(optimizer, config, total_iters, train_loader)
    
    scaler = GradScaler() if config.use_amp else None
    criterion = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
    logger = TrainingLogger(config.log_dir, config.dataset_name)

    best_val_metrics = {'oa': 0.0}
    best_test_metrics = None
    best_model_path = None
    global_step = 0

    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        train_loss, train_acc1, train_metrics, global_step = train_epoch(model, train_loader, optimizer, scheduler, criterion, epoch, config, scaler, global_step)
        val_loss, val_acc1, val_metrics = validate_epoch(model, val_loader, criterion, epoch, config)

        is_best = val_acc1 > best_val_metrics['oa']
        if is_best:
            best_val_metrics = val_metrics.copy()
            
            # Validation 최고 성능을 갱신했을 때, 같은 가중치로 test set 성능 평가
            test_loss, test_acc1, test_metrics = validate_epoch(
                model, test_loader, criterion, epoch, config
            )
            best_test_metrics = test_metrics.copy()
            print(f"[Test @ Best Val] Loss: {test_loss:.4f} | "
                  f"OA: {test_metrics['oa']:.2f}% | "
                  f"AUC: {test_metrics['auc']:.2f}% | "
                  f"F1: {test_metrics['f1']:.2f}%")
        
        saved_path = save_checkpoint(model, epoch, val_acc1, config, is_best)
        if saved_path is not None:
            best_model_path = saved_path

        epoch_time = time.time() - epoch_start_time
        logger.log_epoch(epoch, train_loss, train_metrics, val_loss, val_metrics,
                         optimizer.param_groups[0]['lr'], epoch_time)
        print(f"Epoch {epoch+1} | Train: {train_acc1:.2f}% | "
              f"Val: {val_acc1:.2f}% | Best Val OA: {best_val_metrics['oa']:.2f}%")
            
        if (epoch + 1) % 10 == 0: logger.save_csv()

    logger.save_csv()
    
    # 최종적으로, 논문에 기록할 test 결과 출력 (best validation 기준)
    if best_test_metrics is not None:
        print("[Final Test Result @ Best Val]")
        print(f"OA: {best_test_metrics['oa']:.2f}% | "
              f"AUC: {best_test_metrics['auc']:.2f}% | "
              f"Precision: {best_test_metrics['precision']:.2f}% | "
              f"Sensitivity: {best_test_metrics['sensitivity']:.2f}% | "
              f"Specificity: {best_test_metrics['specificity']:.2f}% | "
              f"F1: {best_test_metrics['f1']:.2f}%")
    else:
        print("[Final Test Result] Best validation 성능이 갱신되지 않아 별도 test 측정을 수행하지 않았습니다.")
    
    # ----- Inference time measurement with best checkpoint + final test OA (CPU 기준) -----
    if best_model_path is not None:
        print(f"[Eval] Best checkpoint 로드 후 inference time 및 test OA 측정 시작: {best_model_path}")
        # PyTorch 2.6 이후 기본 weights_only=True 때문에 발생하는 UnpicklingError를 피하기 위해
        # weights_only=False로 명시적으로 설정하여 전체 checkpoint를 로드합니다. 
        checkpoint = torch.load(best_model_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Inference time과 최종 Test 평가는 항상 CPU 기준으로 수행
        cpu_device = torch.device("cpu")
        model.to(cpu_device)
        model.eval()

        # 최종적으로 best checkpoint 기준 test OA 재측정
        print("[Eval] Best checkpoint 기준 Test 성능 재측정 중...")
        # CPU에서 평가하기 위해 test_loader의 배치를 CPU로 옮기도록 config.device를 임시로 변경
        original_device = config.device
        config.device = cpu_device
        test_loss_final, test_acc1_final, test_metrics_final = validate_epoch(
            model, test_loader, criterion, config.epochs - 1, config
        )
        config.device = original_device
        print(f"[Eval] Final Test OA: {test_metrics_final['oa']:.2f}% | "
              f"AUC: {test_metrics_final['auc']:.2f}% | "
              f"F1: {test_metrics_final['f1']:.2f}%")

        # Inference time 측정 (항상 CPU)
        dummy_input = torch.randn(1, 3, config.image_size, config.image_size).to(cpu_device)
        
        # Warm-up runs (not timed)
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Timed runs
        times = []
        with torch.no_grad():
            for _ in range(10):
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()
                times.append(end - start)
        
        times = np.array(times)
        avg_time_ms = times.mean() * 1000.0
        std_time_ms = times.std() * 1000.0
        print(f"[Eval] Inference time (batch=1, {config.image_size}x{config.image_size}) "
              f"- warmup 10회 후 10회 측정")
        print(f"[Eval] 평균: {avg_time_ms:.3f} ms, 표준편차: {std_time_ms:.3f} ms")
    else:
        print("[Eval] Best checkpoint가 존재하지 않아 inference time을 측정하지 않았습니다.")

if __name__ == "__main__":
    main()
