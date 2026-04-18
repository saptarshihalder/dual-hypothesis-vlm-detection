#!/usr/bin/env python
# score_fusion_training.py
"""
Score-Level Fusion Training: AST + HuBERT (Audio-Only)
=======================================================
Train a Score-Level Fusion head on top of frozen AST + HuBERT,
where BOTH models process mosquito AUDIO.

Architecture:
  - Each backbone produces its own logits (via its classifier head)
  - A learnable weight parameter blends the two softmax scores
  - Final prediction = w_ast * softmax(AST_logits) + w_hub * softmax(HuBERT_logits)
  - The weighting is per-class (vector of weights), trained end-to-end

Dataset: CSV-based (path, label), same format as MoXattn / AST / HuBERT scripts.

Usage:
  python3 AST_HuBERT_score_fusion_training.py --run_precomputation

  # After logits are pre-computed, subsequent runs skip that phase:
  python3 AST_HuBERT_score_fusion_training.py
"""

import os
import argparse
import random
import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
    average_precision_score, f1_score,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — UPDATE THESE PATHS AS NEEDED
# ──────────────────────────────────────────────────────────────────────────────
AUDIO_ROOT     = "/home/tbvl_akshay/[REMOVED]/[REMOVED]/BEANS_data/Audio"
TRAIN_CSV      = "/home/tbvl_akshay/[REMOVED]/[REMOVED]/BEANS_data/metadata/train.csv"
TEST_CSV       = "/home/tbvl_akshay/[REMOVED]/[REMOVED]/BEANS_data/metadata/test.csv"

AST_CKPT       = "/home/tbvl_akshay/[REMOVED]/[REMOVED]/ast_finetuned.pt"
HUB_CKPT       = "/home/tbvl_akshay/[REMOVED]/[REMOVED]/hubert_finetuned_final.pt"
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
HUB_MODEL_DIR  = "/home/tbvl_akshay/[REMOVED]/[REMOVED]/hubert_model"
SAMPLE_RATE    = 16000


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Seed set to {seed}")


# ──────────────────────────────────────────────────────────────────────────────
# Pre-compute LOGITS from both backbones
# ──────────────────────────────────────────────────────────────────────────────
def precompute_logits(
    df: pd.DataFrame,
    audio_root: str,
    feature_root: str,
    ast_model: nn.Module,
    hub_model: nn.Module,
    ast_fe: AutoFeatureExtractor,
    hub_fe: AutoFeatureExtractor,
    sr: int,
    device: torch.device,
):
    """
    Pre-compute AST and HuBERT logits for each audio file.
    Save as (ast_logits, hub_logits) per sample.
    """
    feature_path = Path(feature_root)
    feature_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Pre-computing logits → {feature_root}")

    ast_model.eval()
    hub_model.eval()

    skipped = 0
    saved = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pre-computing logits"):
        basename = os.path.basename(row["path"])
        out_file = feature_path / f"{Path(basename).stem}.pt"
        if out_file.exists():
            saved += 1
            continue

        audio_file = Path(audio_root) / basename
        try:
            info = sf.info(audio_file)
            if not (0 < info.duration <= 60):
                skipped += 1
                continue

            wav, _ = librosa.load(audio_file, sr=sr, duration=60)
            if len(wav) < 400:
                wav = np.pad(wav, (0, 400 - len(wav)))

            a_in = ast_fe([wav], sampling_rate=sr, return_tensors="pt", padding=True)
            h_in = hub_fe([wav], sampling_rate=sr, return_tensors="pt", padding=True)
            a_in = {k: v.to(device) for k, v in a_in.items()}
            h_in = {k: v.to(device) for k, v in h_in.items()}

            with torch.no_grad():
                ast_logits = ast_model(**a_in).logits   # [1, n_cls_ast]
                hub_logits = hub_model(**h_in).logits   # [1, n_cls_hub]

            torch.save((ast_logits.cpu(), hub_logits.cpu()), out_file)
            saved += 1

        except Exception as e:
            logging.warning(f"Skip {audio_file}: {e}")
            skipped += 1

    logging.info(f"Pre-computation done. Saved: {saved}, Skipped: {skipped}")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset & DataLoader
# ──────────────────────────────────────────────────────────────────────────────
class LogitDS(Dataset):
    """Load pre-computed (ast_logits, hub_logits) files."""

    def __init__(self, df: pd.DataFrame, feature_root: str, lbl2idx: Dict[str, int]):
        self.feature_root = Path(feature_root)
        self.lbl2idx = lbl2idx
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        basename = os.path.basename(self.paths[i])
        feature_file = self.feature_root / f"{Path(basename).stem}.pt"
        label = self.lbl2idx[self.labels[i]]
        try:
            ast_logits, hub_logits = torch.load(feature_file)
            return (ast_logits.squeeze(0), hub_logits.squeeze(0)), label
        except FileNotFoundError:
            return None
        except Exception as e:
            return None


def collate_logits(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return (torch.empty(0), torch.empty(0)), torch.empty(0, dtype=torch.long)

    ast_list, hub_list, ys = zip(*[
        (item[0][0], item[0][1], item[1]) for item in batch
    ])
    ast_logits = torch.stack(ast_list)   # [B, n_cls_ast]
    hub_logits = torch.stack(hub_list)   # [B, n_cls_hub]
    ys = torch.tensor(ys, dtype=torch.long)
    return (ast_logits, hub_logits), ys


# ──────────────────────────────────────────────────────────────────────────────
# Score-Level Fusion Head (Learnable Weights)
# ──────────────────────────────────────────────────────────────────────────────
class ScoreFusion(nn.Module):
    """
    Score-level fusion with learnable per-class weights.

    Takes raw logits from AST and HuBERT, converts to probabilities,
    and learns optimal per-class blending weights:
        fused = sigmoid(w_ast) * softmax(ast_logits_mapped)
              + sigmoid(w_hub) * softmax(hub_logits_mapped)

    Includes optional linear mapping if backbone num_labels != n_cls.
    """
    def __init__(self, n_cls_ast: int, n_cls_hub: int, n_cls: int,
                 temperature: float = 1.0):
        super().__init__()
        self.n_cls = n_cls
        self.temperature = temperature

        # Map backbone logits → n_cls if dimensions differ
        self.map_ast = nn.Linear(n_cls_ast, n_cls) if n_cls_ast != n_cls else nn.Identity()
        self.map_hub = nn.Linear(n_cls_hub, n_cls) if n_cls_hub != n_cls else nn.Identity()

        # Learnable per-class fusion weights (initialized to equal weighting)
        self.w_ast = nn.Parameter(torch.zeros(n_cls))  # sigmoid(0) = 0.5
        self.w_hub = nn.Parameter(torch.zeros(n_cls))  # sigmoid(0) = 0.5

    def forward(self, ast_logits: torch.Tensor, hub_logits: torch.Tensor) -> torch.Tensor:
        """
        ast_logits: [B, n_cls_ast]
        hub_logits: [B, n_cls_hub]
        Returns: fused log-probabilities [B, n_cls]
        """
        # Map to common number of classes
        ast_mapped = self.map_ast(ast_logits)  # [B, n_cls]
        hub_mapped = self.map_hub(hub_logits)  # [B, n_cls]

        # Temperature-scaled softmax → probabilities
        p_ast = F.softmax(ast_mapped / self.temperature, dim=1)
        p_hub = F.softmax(hub_mapped / self.temperature, dim=1)

        # Learnable per-class weights (sigmoid ensures [0, 1])
        w_a = torch.sigmoid(self.w_ast)
        w_h = torch.sigmoid(self.w_hub)

        # Weighted combination
        fused = w_a * p_ast + w_h * p_hub  # [B, n_cls]

        # Return log-probabilities (compatible with NLLLoss)
        return torch.log(fused + 1e-8)

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return the current fusion weights for inspection."""
        with torch.no_grad():
            w_a = torch.sigmoid(self.w_ast).cpu().numpy()
            w_h = torch.sigmoid(self.w_hub).cpu().numpy()
        return {'ast_weights': w_a, 'hub_weights': w_h}


# ──────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 7, verbose: bool = True,
                 delta: float = 0.0, path: str = 'checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_acc = -float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, acc: float, model: nn.Module):
        if self.best_score is None or acc > self.best_score + self.delta:
            if self.verbose:
                logging.info(
                    f"Test acc improved ({self.best_acc:.4f} → {acc:.4f}). "
                    f"Saving → {self.path}"
                )
            self.best_score = acc
            self.best_acc = acc
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logging.info(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# ──────────────────────────────────────────────────────────────────────────────
# Testing / Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, dataloader, device, idx2lbl, save_prefix="score_fusion"):
    """Full evaluation with comprehensive metrics."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for (al, hl), y in tqdm(dataloader, desc="Evaluating"):
            if al.nelement() == 0:
                continue
            al, hl, y = al.to(device), hl.to(device), y.to(device)
            log_probs = model(al, hl)
            probs = torch.exp(log_probs)
            preds = log_probs.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().tolist())
            all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)
    n_cls = len(idx2lbl)
    class_names = [idx2lbl[i] for i in range(n_cls)]

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    y_bin = label_binarize(all_labels, classes=list(range(n_cls)))
    try:
        weighted_roc_auc = roc_auc_score(y_bin, all_probs, average='weighted', multi_class='ovr')
    except ValueError:
        weighted_roc_auc = float('nan')
    try:
        macro_roc_auc = roc_auc_score(y_bin, all_probs, average='macro', multi_class='ovr')
    except ValueError:
        macro_roc_auc = float('nan')
    try:
        weighted_pr_auc = average_precision_score(y_bin, all_probs, average='weighted')
    except ValueError:
        weighted_pr_auc = float('nan')
    try:
        macro_pr_auc = average_precision_score(y_bin, all_probs, average='macro')
    except ValueError:
        macro_pr_auc = float('nan')

    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(n_cls)), zero_division=0,
    )
    report_dict = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4, output_dict=True,
    )
    report_str = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4,
    )
    cm = confusion_matrix(all_labels, all_preds)

    per_class = {}
    for i, name in enumerate(class_names):
        mask = [j for j, l in enumerate(all_labels) if l == i]
        if mask:
            correct = sum(1 for j in mask if all_preds[j] == i)
            per_class[name] = {
                'accuracy': correct / len(mask),
                'precision': float(precision_per[i]),
                'recall': float(recall_per[i]),
                'f1': float(f1_per[i]),
                'support': int(support_per[i]),
                'correct': correct,
                'total': len(mask),
            }

    # Print metrics
    logging.info(f"\n{'='*70}")
    logging.info(f"  📊 COMPREHENSIVE TEST METRICS (Score Fusion)")
    logging.info(f"{'='*70}")
    logging.info(f"  Overall Accuracy     : {acc:.4f} ({acc*100:.2f}%)")
    logging.info(f"  Macro F1             : {macro_f1:.4f}")
    logging.info(f"  Weighted F1          : {weighted_f1:.4f}")
    logging.info(f"  Macro ROC AUC        : {macro_roc_auc:.4f}")
    logging.info(f"  Weighted ROC AUC     : {weighted_roc_auc:.4f}")
    logging.info(f"  Macro PR AUC         : {macro_pr_auc:.4f}")
    logging.info(f"  Weighted PR AUC      : {weighted_pr_auc:.4f}")
    logging.info(f"{'='*70}")
    logging.info(f"\n{report_str}")

    logging.info(f"\n  Per-Class Breakdown:")
    for name, m in per_class.items():
        logging.info(
            f"    {name:30s}  Acc={m['accuracy']:.4f}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
            f"F1={m['f1']:.4f}  ({m['correct']}/{m['total']})"
        )

    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Score Fusion Confusion Matrix (AST + HuBERT)')
    plt.tight_layout()
    cm_path = f"{save_prefix}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logging.info(f"Saved confusion matrix → {cm_path}")

    # Save JSON results
    results = {
        'overall_accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'macro_roc_auc': float(macro_roc_auc),
        'weighted_roc_auc': float(weighted_roc_auc),
        'macro_pr_auc': float(macro_pr_auc),
        'weighted_pr_auc': float(weighted_pr_auc),
        'per_class': per_class,
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist(),
    }
    json_path = f"{save_prefix}_test_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logging.info(f"Saved test results → {json_path}")

    # Save predictions CSV
    pred_df = pd.DataFrame({
        'true_label_idx': all_labels,
        'pred_label_idx': all_preds,
        'true_label': [idx2lbl[l] for l in all_labels],
        'pred_label': [idx2lbl[p] for p in all_preds],
        'correct': [int(t == p) for t, p in zip(all_labels, all_preds)],
        'pred_confidence': [float(all_probs[i, p]) for i, p in enumerate(all_preds)],
    })
    for c in range(n_cls):
        pred_df[f'prob_{class_names[c]}'] = all_probs[:, c]
    csv_path = f"{save_prefix}_test_predictions.csv"
    pred_df.to_csv(csv_path, index=False)
    logging.info(f"Saved predictions → {csv_path}")

    # Save summary CSV
    summary_df = pd.DataFrame([{
        'overall_accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'macro_roc_auc': float(macro_roc_auc),
        'weighted_roc_auc': float(weighted_roc_auc),
        'macro_pr_auc': float(macro_pr_auc),
        'weighted_pr_auc': float(weighted_pr_auc),
    }])
    summary_path = f"{save_prefix}_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"Saved metrics summary → {summary_path}")

    return acc, results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ── 1. Load CSVs ─────────────────────────────────────────────────────────
    train_df = pd.read_csv(args.train_csv)
    test_df  = pd.read_csv(args.test_csv)

    classes = sorted(train_df["label"].unique())
    n_cls = len(classes)
    lbl2idx = {c: i for i, c in enumerate(classes)}
    idx2lbl = {i: c for c, i in lbl2idx.items()}
    logging.info(f"Found {n_cls} classes: {classes}")

    # ── 2. Load backbone models ──────────────────────────────────────────────
    logging.info("Loading AST and HuBERT backbones...")

    ast_state = torch.load(args.ast_ckpt, map_location="cpu")
    ast_num_labels = ast_state['classifier.dense.weight'].shape[0]
    hub_state = torch.load(args.hub_ckpt, map_location="cpu")
    hub_num_labels = hub_state['classifier.weight'].shape[0]

    ast_model = ASTForAudioClassification.from_pretrained(
        args.ast_model_name, num_labels=ast_num_labels, ignore_mismatched_sizes=True,
    )
    ast_model.load_state_dict(ast_state)

    hub_model = HubertForSequenceClassification.from_pretrained(
        args.hub_model_dir, num_labels=hub_num_labels, ignore_mismatched_sizes=True,
    )
    hub_model.load_state_dict(hub_state)

    ast_fe = AutoFeatureExtractor.from_pretrained(args.ast_model_name)
    hub_fe = AutoFeatureExtractor.from_pretrained(args.hub_model_dir)

    ast_model.eval().requires_grad_(False)
    hub_model.eval().requires_grad_(False)
    ast_model, hub_model = ast_model.to(device), hub_model.to(device)
    logging.info(f"AST num_labels={ast_num_labels}, HuBERT num_labels={hub_num_labels}")

    # ── 3. Pre-compute logits ────────────────────────────────────────────────
    train_logit_dir = os.path.join(args.feature_root, "train")
    test_logit_dir  = os.path.join(args.feature_root, "test")

    if args.run_precomputation:
        full_df = pd.concat([train_df, test_df]).drop_duplicates(subset=["path"]).reset_index(drop=True)
        precompute_logits(
            full_df, args.audio_root, args.feature_root,
            ast_model, hub_model, ast_fe, hub_fe,
            args.sample_rate, device,
        )
    else:
        logging.info("Skipping pre-computation (use --run_precomputation to run).")

    # Use single feature_root (all logits in one dir, keyed by filename)
    logit_dir = args.feature_root

    # ── 4. Create dataloaders ────────────────────────────────────────────────
    train_ds = LogitDS(train_df, logit_dir, lbl2idx)
    test_ds  = LogitDS(test_df,  logit_dir, lbl2idx)

    train_ld = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_logits, num_workers=4, pin_memory=True,
    )
    test_ld = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_logits, num_workers=4, pin_memory=True,
    )
    logging.info(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # ── 5. Build Score Fusion head ───────────────────────────────────────────
    fusion = ScoreFusion(
        n_cls_ast=ast_num_labels,
        n_cls_hub=hub_num_labels,
        n_cls=n_cls,
        temperature=args.temperature,
    ).to(device)
    logging.info(f"ScoreFusion head: {sum(p.numel() for p in fusion.parameters()):,} params")

    opt = torch.optim.AdamW(
        fusion.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, steps_per_epoch=max(1, len(train_ld)),
        epochs=args.epochs, pct_start=args.warmup_frac,
    )
    loss_fn = nn.NLLLoss()  # model outputs log-probs
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience, verbose=True, path=args.out_ckpt,
    )

    # ── 6. Training loop ─────────────────────────────────────────────────────
    log_data = []
    logging.info("Starting score fusion training...")
    for ep in range(1, args.epochs + 1):
        fusion.train()
        total_train = correct_train = total_loss = 0

        pbar = tqdm(train_ld, desc=f"Epoch {ep}/{args.epochs}")
        for (al, hl), y in pbar:
            if al.nelement() == 0:
                continue
            al, hl, y = al.to(device), hl.to(device), y.to(device)

            log_probs = fusion(al, hl)
            loss = loss_fn(log_probs, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            total_train += y.size(0)
            correct_train += (log_probs.argmax(1) == y).sum().item()
            total_loss += loss.item() * y.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

        tr_acc = correct_train / max(total_train, 1)
        tr_loss = total_loss / max(total_train, 1)

        # Test eval
        fusion.eval()
        total_test = correct_test = 0
        with torch.no_grad():
            for (al, hl), y in test_ld:
                if al.nelement() == 0:
                    continue
                al, hl, y = al.to(device), hl.to(device), y.to(device)
                log_probs = fusion(al, hl)
                total_test += y.size(0)
                correct_test += (log_probs.argmax(1) == y).sum().item()

        test_acc = correct_test / max(total_test, 1)

        weights = fusion.get_weights()
        w_ast_mean = weights['ast_weights'].mean()
        w_hub_mean = weights['hub_weights'].mean()

        logging.info(
            f"Epoch {ep}: Loss={tr_loss:.4f}, "
            f"Train Acc={tr_acc:.4f}, Test Acc={test_acc:.4f}, "
            f"w_ast={w_ast_mean:.3f}, w_hub={w_hub_mean:.3f}"
        )
        log_data.append({
            'epoch': ep, 'train_loss': tr_loss,
            'train_acc': tr_acc, 'test_acc': test_acc,
            'w_ast_mean': float(w_ast_mean), 'w_hub_mean': float(w_hub_mean),
        })

        early_stopping(test_acc, fusion)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered.")
            break

    # ── 7. Load best model & final evaluation ────────────────────────────────
    logging.info(
        f"Training complete. Best Test Acc: {early_stopping.best_acc:.4f}. "
        f"Loading from {args.out_ckpt}"
    )
    fusion.load_state_dict(torch.load(args.out_ckpt, map_location=device))

    # Print final learned weights
    final_weights = fusion.get_weights()
    logging.info("\n" + "="*60)
    logging.info("LEARNED FUSION WEIGHTS (per class)")
    logging.info("="*60)
    for i in range(n_cls):
        logging.info(
            f"  {idx2lbl[i]:30s}  AST={final_weights['ast_weights'][i]:.4f}  "
            f"HuBERT={final_weights['hub_weights'][i]:.4f}"
        )
    logging.info("="*60)

    # Save training log
    log_path = "train_score_fusion.log"
    pd.DataFrame(log_data).to_csv(log_path, index=False)
    logging.info(f"Saved training log → {log_path}")

    # Full test evaluation
    logging.info("Running final evaluation on test set...")
    test_acc, test_results = evaluate_model(
        fusion, test_ld, device, idx2lbl, save_prefix="score_fusion",
    )

    # Save learned weights
    test_results['learned_weights'] = {
        idx2lbl[i]: {
            'ast_weight': float(final_weights['ast_weights'][i]),
            'hub_weight': float(final_weights['hub_weights'][i]),
        }
        for i in range(n_cls)
    }
    with open("score_fusion_learned_weights.json", 'w') as f:
        json.dump(test_results['learned_weights'], f, indent=2)
    logging.info("Saved learned weights → score_fusion_learned_weights.json")

    logging.info(f"\n{'='*70}")
    logging.info(f"  ✅ DONE — Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    logging.info(f"  📁 Checkpoint:        {args.out_ckpt}")
    logging.info(f"  📁 Training log:      {log_path}")
    logging.info(f"  📁 Test results:      score_fusion_test_results.json")
    logging.info(f"  📁 Predictions:       score_fusion_test_predictions.csv")
    logging.info(f"  📁 Metrics summary:   score_fusion_metrics_summary.csv")
    logging.info(f"  📁 Confusion matrix:  score_fusion_confusion_matrix.png")
    logging.info(f"  📁 Learned weights:   score_fusion_learned_weights.json")
    logging.info(f"{'='*70}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Score-Level Fusion Training: AST + HuBERT (Audio-Only)",
    )

    # Paths
    p.add_argument("--audio_root", default=AUDIO_ROOT)
    p.add_argument("--train_csv", default=TRAIN_CSV)
    p.add_argument("--test_csv", default=TEST_CSV)
    p.add_argument("--feature_root", default="./score_fusion_logits",
                    help="Dir to save pre-computed logits")
    p.add_argument("--ast_ckpt", default=AST_CKPT)
    p.add_argument("--hub_ckpt", default=HUB_CKPT)
    p.add_argument("--ast_model_name", default=AST_MODEL_NAME)
    p.add_argument("--hub_model_dir", default=HUB_MODEL_DIR)
    p.add_argument("--out_ckpt", default="score_fusion_best.pt",
                    help="Output checkpoint path")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    p.add_argument("--warmup_frac", type=float, default=0.1)

    # Score Fusion
    p.add_argument("--temperature", type=float, default=1.0,
                    help="Temperature scaling for softmax before fusion")

    # Regularization
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--early_stopping_patience", type=int, default=5)

    # Flags
    p.add_argument("--run_precomputation", action='store_true',
                    help="Run logit pre-computation (needed on first run)")

    args = p.parse_args()
    main(args)


# ──────────────────────────────────────────────────────────────────────────────
# HOW TO RUN
# ──────────────────────────────────────────────────────────────────────────────
# First run (pre-computes logits):
#   python3 multimodal_score_fusion_training.py --run_precomputation
#
# Subsequent runs (skips pre-computation):
#   python3 multimodal_score_fusion_training.py
#
# Custom settings:
#   python3 multimodal_score_fusion_training.py \
#       --run_precomputation \
#       --epochs 50 \
#       --lr 1e-3 \
#       --temperature 2.0
