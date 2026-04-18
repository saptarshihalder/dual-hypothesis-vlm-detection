#!/usr/bin/env python
# pretrained_vit_clip_evaluation.py
"""
Evaluate pretrained ViT + CLIP models with modified classification heads.
Only the classification layer is replaced - no fine-tuning of backbone.
Calculates PR AUC and ROC AUC for each class.
Includes LR scheduler for better training.
"""

import os
import argparse
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    CLIPVisionModel,
    CLIPProcessor,
)
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed: int):
    """Sets the seed for reproducibility for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Seed set to {seed}")

def get_image_paths_and_labels(root_dir: str, lbl2idx: Dict[str, int]) -> List[Tuple[str, int]]:
    """Get all image paths and their corresponding labels from folder structure."""
    image_data = []
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path) and class_name in lbl2idx:
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_data.append((os.path.join(class_path, img_file), lbl2idx[class_name]))
    return image_data

class ImageDataset(Dataset):
    """Dataset for loading images directly."""
    def __init__(self, image_data: List[Tuple[str, int]], processor, model_type='vit'):
        self.image_data = image_data
        self.processor = processor
        self.model_type = model_type

    def __len__(self) -> int:
        return len(self.image_data)

    def __getitem__(self, i: int) -> Optional[Tuple[Dict, int]]:
        img_path, label = self.image_data[i]
        try:
            # Load and process image
            img = Image.open(img_path).convert('RGB')
            
            # Process image based on model type
            if self.model_type == 'vit':
                inputs = self.processor(images=img, return_tensors="pt")
            else:  # CLIP
                inputs = self.processor(images=img, return_tensors="pt")
            
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dim
            
            return inputs, label
            
        except Exception as e:
            logging.warning(f"Could not load image {img_path}. Error: {e}")
            return None

def collate_fn(batch):
    """Custom collate function for image batches."""
    # Filter out failed loads
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    
    inputs, labels = zip(*batch)
    
    # Stack inputs
    batch_inputs = {}
    for key in inputs[0].keys():
        batch_inputs[key] = torch.stack([item[key] for item in inputs])
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    return batch_inputs, labels

class CLIPWithClassifier(nn.Module):
    """CLIP Vision Model with added classifier."""
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(clip_model.config.hidden_size, num_classes)
        
    def forward(self, **inputs):
        outputs = self.clip_model(**inputs)
        pooled_output = outputs.pooler_output  # [batch, hidden_size]
        logits = self.classifier(pooled_output)
        return type('obj', (object,), {'logits': logits})()

def calculate_metrics(y_true, y_probs, class_names):
    """Calculate PR AUC and ROC AUC for each class."""
    n_classes = len(class_names)
    
    # Convert labels to one-hot encoding
    y_true_onehot = label_binarize(y_true, classes=range(n_classes))
    if n_classes == 2:
        y_true_onehot = np.hstack([1-y_true_onehot, y_true_onehot])
    
    pr_auc_scores = []
    roc_auc_scores = []
    
    for i in range(n_classes):
        # PR AUC
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_probs[:, i])
        pr_auc = auc(recall, precision)
        pr_auc_scores.append(pr_auc)
        
        # ROC AUC
        roc_auc = roc_auc_score(y_true_onehot[:, i], y_probs[:, i])
        roc_auc_scores.append(roc_auc)
    
    return pr_auc_scores, roc_auc_scores

def evaluate_model(model, dataloader, class_names, device):
    """Evaluate model and calculate metrics."""
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch[0] is None:  # Skip failed batches
                continue
                
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.array(all_preds)
    
    # Calculate overall accuracy
    accuracy = (all_preds == all_labels).mean()
    
    # Calculate per-class metrics
    pr_auc_scores, roc_auc_scores = calculate_metrics(all_labels, all_probs, class_names)
    
    return accuracy, pr_auc_scores, roc_auc_scores, all_labels, all_preds

def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # ───────── Get classes from folder structure ─────────
    train_classes = [d for d in os.listdir(args.train_root) 
                    if os.path.isdir(os.path.join(args.train_root, d))]
    test_classes = [d for d in os.listdir(args.test_root) 
                   if os.path.isdir(os.path.join(args.test_root, d))]

    # Ensure consistent class ordering
    classes = sorted(list(set(train_classes + test_classes)))
    n_cls = len(classes)
    lbl2idx = {c: i for i, c in enumerate(classes)}
    logging.info(f"Found {n_cls} classes: {classes}")
    
    # ───────── Load Pretrained Models ─────────
    logging.info("Loading pretrained models...")
    
    # Load ViT with modified classification head
    vit = ViTForImageClassification.from_pretrained(args.vit_model_name)
    vit.classifier = nn.Linear(vit.config.hidden_size, n_cls)
    
    # Load CLIP with added classification head
    clip_vision = CLIPVisionModel.from_pretrained(args.clip_model_name)
    clip = CLIPWithClassifier(clip_vision, n_cls)
    
    # Move models to device
    vit.to(device)
    clip.to(device)
    
    # Freeze all parameters except classification layers
    # ViT
    for param in vit.parameters():
        param.requires_grad = False
    for param in vit.classifier.parameters():
        param.requires_grad = True
        
    # CLIP
    for param in clip.clip_model.parameters():
        param.requires_grad = False
    for param in clip.classifier.parameters():
        param.requires_grad = True
    
    logging.info("✅ Only classification layers are trainable")

    # Load processors
    vit_processor = AutoImageProcessor.from_pretrained(args.vit_model_name)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name)
    
    # ───────── Setup Data Loaders ─────────
    train_data = get_image_paths_and_labels(args.train_root, lbl2idx)
    test_data = get_image_paths_and_labels(args.test_root, lbl2idx)
    
    train_ds_vit = ImageDataset(train_data, vit_processor, 'vit')
    test_ds_vit = ImageDataset(test_data, vit_processor, 'vit')
    
    train_ds_clip = ImageDataset(train_data, clip_processor, 'clip')
    test_ds_clip = ImageDataset(test_data, clip_processor, 'clip')

    train_ld_vit = DataLoader(train_ds_vit, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    test_ld_vit = DataLoader(test_ds_vit, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    train_ld_clip = DataLoader(train_ds_clip, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    test_ld_clip = DataLoader(test_ds_clip, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # ───────── Train Classification Layers ─────────
    models = {'ViT': (vit, train_ld_vit, test_ld_vit), 
              'CLIP': (clip, train_ld_clip, test_ld_clip)}
    
    results = {}
    
    for model_name, (model, train_loader, test_loader) in models.items():
        logging.info(f"\n{'='*50}")
        logging.info(f"Training {model_name} Classification Layer")
        logging.info(f"{'='*50}")
        
        # Setup optimizer for classification layer only
        if model_name == 'ViT':
            optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
        else:  # CLIP
            optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
        
        # ✅ ADD LEARNING RATE SCHEDULER
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos'  # Cosine annealing
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for epoch in range(args.epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for batch in pbar:
                if batch[0] is None:
                    continue
                    
                inputs, labels = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(**inputs)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()  # ✅ STEP THE SCHEDULER
                
                total_loss += loss.item()
                predicted = outputs.logits.argmax(dim=-1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # ✅ INCLUDE CURRENT LR IN PROGRESS BAR
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%',
                    'LR': f'{current_lr:.2e}'  # Show current learning rate
                })
        
        # ───────── Evaluate Model ─────────
        logging.info(f"\nEvaluating {model_name}...")
        accuracy, pr_auc_scores, roc_auc_scores, y_true, y_pred = evaluate_model(
            model, test_loader, classes, device
        )
        
        # Store results
        results[model_name] = {
            'accuracy': accuracy,
            'pr_auc_scores': pr_auc_scores,
            'roc_auc_scores': roc_auc_scores,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        # Print results
        logging.info(f"\n{model_name} Results:")
        logging.info(f"Overall Accuracy: {accuracy:.4f}")
        logging.info(f"\nPer-Class Metrics:")
        
        metrics_df = []
        for i, class_name in enumerate(classes):
            logging.info(f"Class {class_name}:")
            logging.info(f"  PR AUC: {pr_auc_scores[i]:.4f}")
            logging.info(f"  ROC AUC: {roc_auc_scores[i]:.4f}")
            
            metrics_df.append({
                'Class': class_name,
                'PR_AUC': pr_auc_scores[i],
                'ROC_AUC': roc_auc_scores[i]
            })
        
        # Save detailed results
        metrics_df = pd.DataFrame(metrics_df)
        metrics_df.to_csv(f"{model_name.lower()}_class_metrics.csv", index=False)
        
        # Calculate and print average scores
        avg_pr_auc = np.mean(pr_auc_scores)
        avg_roc_auc = np.mean(roc_auc_scores)
        logging.info(f"\nAverage PR AUC: {avg_pr_auc:.4f}")
        logging.info(f"Average ROC AUC: {avg_roc_auc:.4f}")
        logging.info(f"Final Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
    
    # ───────── Compare Models ─────────
    logging.info(f"\n{'='*60}")
    logging.info("MODEL COMPARISON")
    logging.info(f"{'='*60}")
    
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'Avg_PR_AUC': np.mean(result['pr_auc_scores']),
            'Avg_ROC_AUC': np.mean(result['roc_auc_scores'])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv("model_comparison.csv", index=False)
    
    logging.info("\nModel Comparison:")
    logging.info(comparison_df.to_string(index=False))
    
    logging.info(f"\n✅ Results saved to CSV files")
    logging.info(f"📊 Individual class metrics: vit_class_metrics.csv, clip_class_metrics.csv")
    logging.info(f"📈 Model comparison: model_comparison.csv")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate Pretrained ViT + CLIP with Modified Classification Heads")
    
    # --- Data Paths ---
    p.add_argument("--train_root", required=True, help="Root directory of training images with class subfolders.")
    p.add_argument("--test_root", required=True, help="Root directory of test images with class subfolders.")
    p.add_argument("--vit_model_name", default="google/vit-base-patch16-224")
    p.add_argument("--clip_model_name", default="openai/clip-vit-base-patch32")
    
    # --- Training Hyperparameters ---
    p.add_argument("--epochs", type=int, default=5, help="Number of epochs to train classification layer.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.001, help="Learning rate for classification layer.")
    
    # --- Scheduler Parameters ---
    p.add_argument("--scheduler", default="onecycle", choices=["none", "onecycle", "cosine"], help="Learning rate scheduler type.")
    p.add_argument("--warmup_pct", type=float, default=0.1, help="Warmup percentage for OneCycleLR.")
    
    args = p.parse_args()
    main(args)



# python pretrained_vit_clip_evaluation.py \
#     --train_root /home/tbvl_akshay/[REMOVED]/[REMOVED]/Advanced-Vision-Transformers-and-Open-Set-Learning-for-Mosquito-Classification-/Dataset/trainset \
#     --test_root /home/tbvl_akshay/[REMOVED]/[REMOVED]/Advanced-Vision-Transformers-and-Open-Set-Learning-for-Mosquito-Classification-/Dataset/testset \
#     --batch_size 8\
#     --epochs 20\
#     --lr 0.001
