#!/usr/bin/env python
# score_level_fusion_eval.py
"""
Score-level fusion evaluation of pretrained AST + HuBERT models.
No training required - just average the prediction scores with comprehensive metrics.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoFeatureExtractor,
    ASTForAudioClassification,
    HubertForSequenceClassification,
)
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    classification_report
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration - UPDATE THESE PATHS
TEST_CSV = "/home/tbvl_akshay/[REMOVED]/[REMOVED]/BEANS_data/metadata/test.csv"
AUDIO_ROOT = "/home/tbvl_akshay/[REMOVED]/[REMOVED]/BEANS_data/Audio"
AST_CKPT = "/home/tbvl_akshay/[REMOVED]/[REMOVED]/ast_new_dataset_final.pt"
HUB_CKPT = "/NAS_DISK/best_hubert_new_dataset.pt"
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
HUB_MODEL_DIR = "/home/tbvl_akshay/[REMOVED]/[REMOVED]/hubert_model"
BATCH_SIZE = 4
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class AudioDS(Dataset):
    """Dataset to load raw audio files."""
    def __init__(self, df: pd.DataFrame, audio_root: str, lbl2idx: Dict[str, int], sr: int):
        self.paths = df["path"].map(os.path.basename).tolist()
        self.labels = df["label"].tolist()
        self.audio_root = Path(audio_root)
        self.lbl2idx = lbl2idx
        self.sr = sr

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        wav_path = self.audio_root / self.paths[i]
        label = self.lbl2idx[self.labels[i]]
        
        try:
            # Check audio duration
            info = sf.info(wav_path)
            if not (0 < info.duration <= 60):
                return None
                
            # Load audio (up to 60 seconds)
            audio, _ = librosa.load(wav_path, sr=self.sr, duration=60.0)
            if len(audio) < 400:
                audio = np.pad(audio, (0, 400 - len(audio)))
                
            return audio, label, self.paths[i]
        except Exception as e:
            logging.warning(f"Could not load {wav_path}: {e}")
            return None

def collate_audio_fn(batch, feature_extractor):
    """Collate function for audio data."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None
        
    audios, labels, names = zip(*batch)
    
    # Process audio through feature extractor
    inputs = feature_extractor(
        list(audios), 
        sampling_rate=feature_extractor.sampling_rate, 
        return_tensors="pt", 
        padding=True
    )
    
    return inputs, torch.tensor(labels), list(names)

def load_and_filter_test_csv(test_csv_path):
    """
    Load test CSV and filter for only the 3 relevant classes.
    Map test CSV labels to training class names.
    """
    logging.info(f"Loading and filtering test CSV: {test_csv_path}")
    
    # Load test CSV
    df_test = pd.read_csv(test_csv_path)
    df_test['label'] = df_test['label'].str.strip()
    
    logging.info(f"Original test CSV has {len(df_test)} samples")
    logging.info(f"Original test classes: {df_test['label'].unique().tolist()}")
    
    # Define relevant classes from test CSV
    relevant_test_classes = ['ae aegypti', 'an dirus', 'culex quinquefasciatus']
    
    # Filter for only relevant classes
    df_test_filtered = df_test[df_test['label'].isin(relevant_test_classes)].copy()
    
    # Map test CSV labels to standardized training labels
    test_to_train_label_mapping = {
        'ae aegypti': 'ae aegypti',
        'an dirus': 'an dirus', 
        'culex quinquefasciatus': 'culex quinquefasciatus'
    }
    
    # Apply mapping
    df_test_filtered['mapped_label'] = df_test_filtered['label'].map(test_to_train_label_mapping)
    
    # Remove any unmapped labels
    df_test_filtered = df_test_filtered.dropna(subset=['mapped_label'])
    
    # Create final test DataFrame
    df_test_final = df_test_filtered[['path', 'mapped_label']].copy()
    df_test_final.columns = ['path', 'label']
    
    logging.info(f"Filtered test CSV has {len(df_test_final)} samples from 3 classes")
    logging.info(f"Test samples per class:")
    for class_label, count in df_test_final['label'].value_counts().items():
        logging.info(f"  {class_label}: {count}")
    
    return df_test_final

def calculate_weighted_metrics(y_true, y_pred, y_prob, class_names):
    """
    Calculate weighted metrics: Accuracy, F1, ROC-AUC, and PR-AUC
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # For multiclass AUC metrics, we need to binarize the labels
    n_classes = len(class_names)
    
    # Weighted ROC-AUC
    try:
        weighted_roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except Exception as e:
        logging.warning(f"Could not calculate weighted ROC-AUC: {e}")
        weighted_roc_auc = np.nan
    
    # Weighted PR-AUC
    try:
        # Convert labels to binary format for PR-AUC calculation
        y_true_binary = label_binarize(y_true, classes=range(n_classes))
        weighted_pr_auc = average_precision_score(y_true_binary, y_prob, average='weighted')
    except Exception as e:
        logging.warning(f"Could not calculate weighted PR-AUC: {e}")
        weighted_pr_auc = np.nan
    
    return {
        'accuracy': accuracy,
        'weighted_f1': weighted_f1,
        'weighted_roc_auc': weighted_roc_auc,
        'weighted_pr_auc': weighted_pr_auc
    }

def evaluate_single_model(model, dataloader, device, model_name, classes):
    """Evaluate a single model and return predictions and probabilities."""
    logging.info(f"🔍 Evaluating {model_name}...")
    
    model.eval().to(device)
    all_preds = []
    all_probs = []
    all_labels = []
    all_names = []
    
    with torch.no_grad():
        for inputs, labels, names in tqdm(dataloader, desc=f"Evaluating {model_name}"):
            if inputs is None:
                continue
                
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            # Forward pass
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1)  # Convert to probabilities
            preds = torch.argmax(logits, dim=1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_names.extend(names)
    
    # Concatenate probabilities
    all_probs = np.vstack(all_probs) if all_probs else np.array([])
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate comprehensive metrics
    metrics = calculate_weighted_metrics(all_labels, all_preds, all_probs, classes)
    
    return all_preds, all_probs, all_labels, all_names, metrics

def score_level_fusion(ast_probs, hub_probs, fusion_weights=(0.5, 0.5)):
    """
    Perform score-level fusion of AST and HuBERT predictions.
    
    Args:
        ast_probs: AST probability predictions [N, num_classes]
        hub_probs: HuBERT probability predictions [N, num_classes]
        fusion_weights: Weights for (AST, HuBERT) - default is equal weighting
    
    Returns:
        fused_probs: Fused probability predictions [N, num_classes]
        fused_preds: Final predicted classes [N]
    """
    # Weighted average of probabilities
    fused_probs = (fusion_weights[0] * ast_probs + 
                   fusion_weights[1] * hub_probs)
    
    # Get final predictions
    fused_preds = np.argmax(fused_probs, axis=1)
    
    return fused_probs, fused_preds

def evaluate_and_report_comprehensive(y_true, y_pred, y_prob, model_name, classes):
    """Calculate and display comprehensive evaluation metrics."""
    # Calculate all metrics
    metrics = calculate_weighted_metrics(y_true, y_pred, y_prob, classes)
    
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} RESULTS")
    print(f"{'='*60}")
    print(f"🎯 Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"📈 Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    print(f"🎯 Weighted ROC-AUC: {metrics['weighted_roc_auc']:.4f}")
    print(f"📊 Weighted PR-AUC: {metrics['weighted_pr_auc']:.4f}")
    print(f"📊 Number of samples: {len(y_true)}")
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print(f"\n📋 Classification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🔢 Confusion Matrix:")
    print(cm)
    
    # Per-class accuracy
    print(f"\n📈 Per-Class Accuracy:")
    per_class_acc = {}
    for i, class_name in enumerate(classes):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == i).mean()
            class_count = class_mask.sum()
            per_class_acc[class_name] = {
                'accuracy': float(class_acc),
                'correct': int(class_acc * class_count),
                'total': int(class_count)
            }
            print(f"  {class_name}: {class_acc:.4f} ({int(class_acc * class_count)}/{class_count})")
    
    return {
        'metrics': metrics,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'per_class_accuracy': per_class_acc,
        'predictions': y_pred.tolist(),
        'probabilities': y_prob.tolist() if len(y_prob) > 0 else []
    }

def save_confusion_matrix(cm, class_names, model_name, filename):
    """Save confusion matrix plot."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} - Test Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    logging.info("🚀 Starting Score-Level Fusion Evaluation")
    logging.info(f"Using device: {DEVICE}")
    
    # ───────── Load Test Data ─────────
    test_df = load_and_filter_test_csv(TEST_CSV)
    classes = sorted(test_df["label"].unique())
    n_cls = len(classes)
    lbl2idx = {c: i for i, c in enumerate(classes)}
    
    logging.info(f"📊 Test dataset: {len(test_df)} samples, {n_cls} classes")
    logging.info(f"Classes: {classes}")
    
    # ───────── Load Pretrained Models ─────────
    logging.info("🔧 Loading pretrained models...")
    
    # Load AST
    ast_checkpoint = torch.load(AST_CKPT, map_location="cpu")
    # Determine original number of classes from checkpoint
    original_ast_classes = 3  # Assuming 3 classes based on your training
    
    ast_model = ASTForAudioClassification.from_pretrained(
        AST_MODEL_NAME, 
        num_labels=original_ast_classes, 
        ignore_mismatched_sizes=True
    )
    ast_model.load_state_dict(ast_checkpoint)
    ast_fe = AutoFeatureExtractor.from_pretrained(AST_MODEL_NAME)
    
    # Load HuBERT
    hub_checkpoint = torch.load(HUB_CKPT, map_location="cpu")
    hub_model = HubertForSequenceClassification.from_pretrained(HUB_MODEL_DIR)
    
    # Set HuBERT configuration (same as training)
    hub_model.hubert._mask_hidden_states = lambda hidden_states, mask_time_indices=None, attention_mask=None: hidden_states
    hub_model.config.mask_time_prob = 0.0
    hub_model.config.mask_time_length = 0
    hub_model.config.mask_time_min_masks = 0
    hub_model.config.mask_feature_prob = 0.0
    hub_model.config.apply_spec_augment = False
    
    # Set correct classifier for HuBERT (3 classes, 256 input features)
    hub_model.classifier = torch.nn.Linear(256, n_cls)
    hub_model.load_state_dict(hub_checkpoint)
    
    hub_fe = AutoFeatureExtractor.from_pretrained(HUB_MODEL_DIR)
    
    logging.info("✅ Models loaded successfully!")
    
    # ───────── Prepare Data Loaders ─────────
    audio_ds = AudioDS(test_df, AUDIO_ROOT, lbl2idx, SAMPLE_RATE)
    
    # Separate loaders for each model (different feature extractors)
    ast_loader = DataLoader(
        audio_ds, 
        batch_size=BATCH_SIZE, 
        collate_fn=lambda b: collate_audio_fn(b, ast_fe),
        num_workers=2, 
        pin_memory=True
    )
    
    hub_loader = DataLoader(
        audio_ds, 
        batch_size=BATCH_SIZE, 
        collate_fn=lambda b: collate_audio_fn(b, hub_fe),
        num_workers=2, 
        pin_memory=True
    )
    
    # ───────── Evaluate Individual Models ─────────
    logging.info(f"\n{'='*60}")
    logging.info("INDIVIDUAL MODEL EVALUATION")
    logging.info(f"{'='*60}")
    
    # Evaluate AST
    ast_preds, ast_probs, ast_labels, ast_names, ast_metrics = evaluate_single_model(
        ast_model, ast_loader, DEVICE, "AST", classes
    )
    ast_results = evaluate_and_report_comprehensive(ast_labels, ast_preds, ast_probs, "AST", classes)
    
    # Evaluate HuBERT
    hub_preds, hub_probs, hub_labels, hub_names, hub_metrics = evaluate_single_model(
        hub_model, hub_loader, DEVICE, "HuBERT", classes
    )
    hub_results = evaluate_and_report_comprehensive(hub_labels, hub_preds, hub_probs, "HuBERT", classes)
    
    # ───────── Score-Level Fusion ─────────
    logging.info(f"\n{'='*60}")
    logging.info("SCORE-LEVEL FUSION EVALUATION")
    logging.info(f"{'='*60}")
    
    # Ensure same order of samples (should be the same since same dataset)
    assert np.array_equal(ast_labels, hub_labels), "Label mismatch between models!"
    
    # Perform fusion with different weight combinations
    fusion_configs = [
        ("Equal Weight", (0.5, 0.5)),
        ("AST Weighted", (0.7, 0.3)),
        ("HuBERT Weighted", (0.3, 0.7)),
    ]
    
    best_fusion_acc = 0
    best_fusion_config = None
    all_fusion_results = {}
    
    for config_name, weights in fusion_configs:
        logging.info(f"\n🔀 Testing fusion: {config_name} {weights}")
        
        # Perform fusion
        fused_probs, fused_preds = score_level_fusion(ast_probs, hub_probs, weights)
        
        # Evaluate fusion
        fusion_results = evaluate_and_report_comprehensive(
            ast_labels, fused_preds, fused_probs, 
            f"Score-Fusion-{config_name}", classes
        )
        
        # Track best fusion
        fusion_acc = fusion_results['metrics']['accuracy']
        if fusion_acc > best_fusion_acc:
            best_fusion_acc = fusion_acc
            best_fusion_config = config_name
            best_fusion_results = fusion_results
        
        all_fusion_results[config_name] = fusion_results
    
    # ───────── Final Comparison ─────────
    logging.info(f"\n{'='*60}")
    logging.info("FINAL COMPREHENSIVE COMPARISON")
    logging.info(f"{'='*60}")
    
    results_summary = {
        'AST': {
            'accuracy': ast_results['metrics']['accuracy'],
            'weighted_f1': ast_results['metrics']['weighted_f1'],
            'weighted_roc_auc': ast_results['metrics']['weighted_roc_auc'],
            'weighted_pr_auc': ast_results['metrics']['weighted_pr_auc']
        },
        'HuBERT': {
            'accuracy': hub_results['metrics']['accuracy'],
            'weighted_f1': hub_results['metrics']['weighted_f1'],
            'weighted_roc_auc': hub_results['metrics']['weighted_roc_auc'],
            'weighted_pr_auc': hub_results['metrics']['weighted_pr_auc']
        },
        'Best_Fusion': {
            'accuracy': best_fusion_acc,
            'weighted_f1': best_fusion_results['metrics']['weighted_f1'],
            'weighted_roc_auc': best_fusion_results['metrics']['weighted_roc_auc'],
            'weighted_pr_auc': best_fusion_results['metrics']['weighted_pr_auc']
        }
    }
    
    print(f"\n📊 COMPREHENSIVE METRICS SUMMARY:")
    for model_name, metrics in results_summary.items():
        print(f"\n{model_name}:")
        print(f"  ✅ Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  📈 Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"  🎯 Weighted ROC-AUC: {metrics['weighted_roc_auc']:.4f}")
        print(f"  📊 Weighted PR-AUC: {metrics['weighted_pr_auc']:.4f}")
    
    best_model = max(results_summary.keys(), key=lambda k: results_summary[k]['accuracy'])
    print(f"\n🏆 Best performing approach: {best_model}")
    print(f"🔀 Best fusion configuration: {best_fusion_config}")
    
    # ───────── Save Results and Visualizations ─────────
    # Save confusion matrices
    save_confusion_matrix(
        np.array(ast_results['confusion_matrix']), classes, "AST", 
        "ast_score_fusion_confusion_matrix.png"
    )
    save_confusion_matrix(
        np.array(hub_results['confusion_matrix']), classes, "HuBERT", 
        "hubert_score_fusion_confusion_matrix.png"
    )
    save_confusion_matrix(
        np.array(best_fusion_results['confusion_matrix']), classes, 
        f"Score-Fusion-{best_fusion_config}", "best_score_fusion_confusion_matrix.png"
    )
    
    # Save comprehensive results
    final_results = {
        'model_type': 'Score_Level_Fusion_Comprehensive',
        'individual_models': {
            'AST': ast_results,
            'HuBERT': hub_results
        },
        'fusion_results': all_fusion_results,
        'summary': results_summary,
        'best_fusion_config': best_fusion_config,
        'classes': classes,
        'num_test_samples': len(ast_labels),
        'configuration': {
            'ast_checkpoint': AST_CKPT,
            'hub_checkpoint': HUB_CKPT,
            'test_csv': TEST_CSV,
            'batch_size': BATCH_SIZE,
            'sample_rate': SAMPLE_RATE
        }
    }
    
    with open("comprehensive_score_fusion_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    
    logging.info(f"\n💾 Results saved:")
    logging.info(f"  - comprehensive_score_fusion_results.json")
    logging.info(f"  - ast_score_fusion_confusion_matrix.png")
    logging.info(f"  - hubert_score_fusion_confusion_matrix.png")
    logging.info(f"  - best_score_fusion_confusion_matrix.png")
    
    logging.info(f"\n✅ Score-level fusion evaluation completed!")
    logging.info(f"🎯 Best Accuracy: {best_fusion_acc:.4f} ({best_fusion_acc*100:.2f}%)")
    logging.info(f"🔀 Best Configuration: {best_fusion_config}")
    
    return results_summary

if __name__ == "__main__":
    main()
