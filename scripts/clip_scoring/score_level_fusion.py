#!/usr/bin/env python
# score_fusion_eval.py
"""
Score-level fusion evaluation of pretrained AST + HuBERT models.
No training required - just average the prediction scores.
"""

import os
import argparse
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import json

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def evaluate_single_model(model, dataloader, device, model_name):
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
    all_probs = np.vstack(all_probs)
    
    return np.array(all_preds), all_probs, np.array(all_labels), all_names

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

def evaluate_and_report(y_true, y_pred, y_prob, model_name, classes):
    """Calculate and display evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} RESULTS")
    print(f"{'='*60}")
    print(f"🎯 Test Accuracy: {accuracy*100:.2f}%")
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
    for i, class_name in enumerate(classes):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == i).mean()
            class_count = class_mask.sum()
            print(f"  {class_name}: {class_acc:.4f} ({int(class_acc * class_count)}/{class_count})")
    
    return {
        'accuracy': float(accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'probabilities': y_prob.tolist()
    }

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # ───────── Load Test Data ─────────
    test_df = pd.read_csv(args.test_csv)
    classes = sorted(test_df["label"].unique())
    n_cls = len(classes)
    lbl2idx = {c: i for i, c in enumerate(classes)}
    
    logging.info(f"📊 Test dataset: {len(test_df)} samples, {n_cls} classes")
    logging.info(f"Classes: {classes}")
    
    # ───────── Load Pretrained Models ─────────
    logging.info("🔧 Loading pretrained models...")
    
    # Load AST
    ast_model = ASTForAudioClassification.from_pretrained(
        args.ast_model_name, 
        num_labels=n_cls, 
        ignore_mismatched_sizes=True
    )
    ast_model.load_state_dict(torch.load(args.ast_ckpt, map_location="cpu"))
    ast_fe = AutoFeatureExtractor.from_pretrained(args.ast_model_name)
    
    # Load HuBERT
    hub_model = HubertForSequenceClassification.from_pretrained(
        args.hub_model_dir, 
        num_labels=n_cls, 
        ignore_mismatched_sizes=True
    )
    hub_model.load_state_dict(torch.load(args.hub_ckpt, map_location="cpu"))
    hub_fe = AutoFeatureExtractor.from_pretrained(args.hub_model_dir)
    
    logging.info("✅ Models loaded successfully!")
    
    # ───────── Prepare Data Loaders ─────────
    audio_ds = AudioDS(test_df, args.audio_root, lbl2idx, args.sample_rate)
    
    # Separate loaders for each model (different feature extractors)
    ast_loader = DataLoader(
        audio_ds, 
        batch_size=args.batch_size, 
        collate_fn=lambda b: collate_audio_fn(b, ast_fe),
        num_workers=4, 
        pin_memory=True
    )
    
    hub_loader = DataLoader(
        audio_ds, 
        batch_size=args.batch_size, 
        collate_fn=lambda b: collate_audio_fn(b, hub_fe),
        num_workers=4, 
        pin_memory=True
    )
    
    # ───────── Evaluate Individual Models ─────────
    logging.info(f"\n{'='*60}")
    logging.info("INDIVIDUAL MODEL EVALUATION")
    logging.info(f"{'='*60}")
    
    # Evaluate AST
    ast_preds, ast_probs, ast_labels, ast_names = evaluate_single_model(
        ast_model, ast_loader, device, "AST"
    )
    ast_results = evaluate_and_report(ast_labels, ast_preds, ast_probs, "AST", classes)
    
    # Evaluate HuBERT
    hub_preds, hub_probs, hub_labels, hub_names = evaluate_single_model(
        hub_model, hub_loader, device, "HuBERT"
    )
    hub_results = evaluate_and_report(hub_labels, hub_preds, hub_probs, "HuBERT", classes)
    
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
        fusion_results = evaluate_and_report(
            ast_labels, fused_preds, fused_probs, 
            f"Fusion-{config_name}", classes
        )
        
        # Track best fusion
        fusion_acc = fusion_results['accuracy']
        if fusion_acc > best_fusion_acc:
            best_fusion_acc = fusion_acc
            best_fusion_config = config_name
        
        all_fusion_results[config_name] = fusion_results
    
    # ───────── Final Comparison ─────────
    logging.info(f"\n{'='*60}")
    logging.info("FINAL COMPARISON")
    logging.info(f"{'='*60}")
    
    results_summary = {
        'AST': ast_results['accuracy'],
        'HuBERT': hub_results['accuracy'],
        'Best Fusion': best_fusion_acc
    }
    
    print(f"\n📊 ACCURACY SUMMARY:")
    for model_name, accuracy in results_summary.items():
        print(f"  {model_name}: {accuracy*100:.2f}%")
    
    print(f"\n🏆 Best performing approach: {max(results_summary, key=results_summary.get)}")
    print(f"🔀 Best fusion configuration: {best_fusion_config}")
    
    # ───────── Save Results ─────────
    final_results = {
        'individual_models': {
            'AST': ast_results,
            'HuBERT': hub_results
        },
        'fusion_results': all_fusion_results,
        'summary': results_summary,
        'best_fusion_config': best_fusion_config,
        'classes': classes,
        'num_test_samples': len(ast_labels)
    }
    
    with open("score_fusion_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    
    logging.info(f"\n💾 Results saved to: score_fusion_results.json")
    logging.info(f"✅ Score-level fusion evaluation completed!")
    
    return results_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score-Level Fusion Evaluation of AST + HuBERT")
    
    # Data paths
    parser.add_argument("--test_csv", required=True, help="Path to test CSV file")
    parser.add_argument("--audio_root", required=True, help="Root directory of audio files")
    
    # Model paths
    parser.add_argument("--ast_model_name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--hub_model_dir", required=True, help="Path to local HuBERT model directory")
    parser.add_argument("--ast_ckpt", required=True, help="Path to fine-tuned AST checkpoint")
    parser.add_argument("--hub_ckpt", required=True, help="Path to fine-tuned HuBERT checkpoint")
    
    # Evaluation parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    
    args = parser.parse_args()
    main(args)


# python score_level_fusion.py \
#     --test_csv /home/[REMOVED]/Mosquito/BEANS_data/metadata/test.csv \
#     --audio_root /home/[REMOVED]/Mosquito/BEANS_data/Audio \
#     --ast_ckpt /home/[REMOVED]/Mosquito/ast_finetuned.pt \
#     --hub_ckpt /home/[REMOVED]/Mosquito/hubert_finetuned_final.pt \
#     --hub_model_dir /home/[REMOVED]/Mosquito/hubert_model \
#     --batch_size 4
