#!/usr/bin/env python3
"""
Dual-Hypothesis Pipeline: Steps 1-5
CLIP extraction -> Discrepancy -> Classifier -> Soft labels
"""

import os, json, glob, time, sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

RESULTS_DIR    = "/NAS_DISK/Saptarshi_data/results"
BACKUP_DIR     = "/NAS_DISK/Saptarshi_data/results_backup"
DATASET_DIR    = "/NAS_DISK/Saptarshi_data/dataset"
HF_CACHE       = "/NAS_DISK/Saptarshi_data/hf_cache"
OUTPUT_DIR     = "/NAS_DISK/Saptarshi_data/pipeline_output"
REAL_IMG_DIR   = os.path.join(DATASET_DIR, "real", "coco")
FAKE_IMG_DIR   = os.path.join(DATASET_DIR, "fake", "midjourney")

N_REAL = 1500
N_FAKE = 1500
CLIP_BATCH = 64
SEED = 42

os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE


def load_all_vlm_results():
    all_results = []
    search_paths = [RESULTS_DIR, BACKUP_DIR]
    json_files = []
    for d in search_paths:
        if os.path.isdir(d):
            json_files.extend(glob.glob(os.path.join(d, "*.json")))
    print(f"Found {len(json_files)} JSON files")
    for fp in json_files:
        try:
            with open(fp) as f:
                data = json.load(f)
            entries = data if isinstance(data, list) else data.get("results", [])
            if entries:
                print(f"  {Path(fp).name}: {len(entries)} entries")
                all_results.extend(entries)
        except Exception as e:
            print(f"  Skip {fp}: {e}")
    print(f"\nTotal VLM entries loaded: {len(all_results)}")
    by_image = defaultdict(list)
    models_seen = set()
    for r in all_results:
        img_id = r.get("image_id", "")
        by_image[img_id].append(r)
        models_seen.add(r.get("model", "unknown"))
    print(f"Unique images: {len(by_image)}")
    print(f"Models: {models_seen}")
    return by_image, models_seen


def select_3k_subset(by_image):
    np.random.seed(SEED)
    real_ids, fake_ids = [], []
    for img_id, entries in by_image.items():
        gt = entries[0].get("ground_truth", "").upper()
        n_models = len(set(e.get("model", "") for e in entries))
        if n_models < 3:
            continue
        if gt == "REAL":
            real_ids.append(img_id)
        elif gt == "FAKE":
            fake_ids.append(img_id)
    print(f"\nEligible: {len(real_ids)} real, {len(fake_ids)} fake")
    np.random.shuffle(real_ids)
    np.random.shuffle(fake_ids)
    selected_real = real_ids[:N_REAL]
    selected_fake = fake_ids[:N_FAKE]
    print(f"Selected: {len(selected_real)} real, {len(selected_fake)} fake")
    return selected_real + selected_fake


def find_image_path(img_id):
    for d in [REAL_IMG_DIR, FAKE_IMG_DIR]:
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            p = os.path.join(d, img_id + ext)
            if os.path.exists(p):
                return p
    for d in [REAL_IMG_DIR, FAKE_IMG_DIR]:
        if os.path.isdir(d):
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                matches = glob.glob(os.path.join(d, "**", img_id + ext), recursive=True)
                if matches:
                    return matches[0]
    return None


def clip_extract(selected_ids, by_image):
    import torch
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", cache_dir=HF_CACHE
        )
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        use_open_clip = True
    except ImportError:
        import clip
        model, preprocess = clip.load("ViT-L-14", device="cuda")
        tokenizer = clip.tokenize
        use_open_clip = False

    model = model.cuda().eval()
    print(f"\nCLIP ViT-L-14 loaded")

    from PIL import Image

    image_features = {}
    text_scores = {}

    print("\nEncoding images...")
    batch_paths, batch_ids = [], []
    for i, img_id in enumerate(selected_ids):
        path = find_image_path(img_id)
        if path is None:
            continue
        batch_paths.append(path)
        batch_ids.append(img_id)
        if len(batch_paths) == CLIP_BATCH or i == len(selected_ids) - 1:
            images = []
            valid_ids = []
            for p, iid in zip(batch_paths, batch_ids):
                try:
                    img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                    images.append(img)
                    valid_ids.append(iid)
                except:
                    pass
            if images:
                images_t = torch.cat(images).cuda()
                with torch.no_grad(), torch.cuda.amp.autocast():
                    feats = model.encode_image(images_t)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                for j, iid in enumerate(valid_ids):
                    image_features[iid] = feats[j].cpu().numpy()
            batch_paths, batch_ids = [], []
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(selected_ids)} images encoded")
    print(f"  Encoded {len(image_features)} images")

    print("\nScoring texts against images...")
    count = 0
    for img_id in selected_ids:
        if img_id not in image_features:
            continue
        img_feat = torch.tensor(image_features[img_id]).unsqueeze(0).cuda()
        for entry in by_image[img_id]:
            model_name = entry.get("model", "unknown")
            assumption = entry.get("assumption", "")
            caption = entry.get("caption", "")
            cues = [entry.get(f"cue_{i}", "") for i in range(1, 4)]
            texts = [t if t else "no description" for t in [caption] + cues]
            try:
                if use_open_clip:
                    tokens = tokenizer(texts).cuda()
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        text_feats = model.encode_text(tokens)
                        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                else:
                    tokens = tokenizer(texts, truncate=True).cuda()
                    with torch.no_grad():
                        text_feats = model.encode_text(tokens)
                        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                sims = (img_feat @ text_feats.T).squeeze(0).cpu().numpy()
                key = (img_id, model_name, assumption)
                text_scores[key] = {
                    "caption_score": float(sims[0]),
                    "cue1_score": float(sims[1]),
                    "cue2_score": float(sims[2]),
                    "cue3_score": float(sims[3]),
                    "mean_cue_score": float(np.mean(sims[1:4])),
                    "mean_all_score": float(np.mean(sims)),
                }
                count += 1
            except:
                pass
        if count % 2000 == 0 and count > 0:
            print(f"  {count} text-image pairs scored")
    print(f"  Total scored: {count}")
    return image_features, text_scores


def compute_features(selected_ids, by_image, text_scores, models_seen):
    feature_names = []
    for m in sorted(models_seen):
        short = m.replace(".", "").replace("-", "")[:10]
        for metric in ["caption", "mean_cue", "mean_all"]:
            feature_names.append(f"{short}_{metric}_disc")
            feature_names.append(f"{short}_{metric}_real")
            feature_names.append(f"{short}_{metric}_fake")
    feature_names += [
        "avg_caption_disc", "avg_cue_disc", "avg_all_disc",
        "std_caption_disc", "std_cue_disc", "std_all_disc",
        "max_disc", "min_disc", "disc_range",
        "avg_real_score", "avg_fake_score",
        "n_models_agree_direction",
    ]
    X, y, ids = [], [], []
    for img_id in selected_ids:
        entries = by_image[img_id]
        gt = entries[0].get("ground_truth", "").upper()
        label = 0 if gt == "REAL" else 1
        per_model_discs = {"caption": [], "mean_cue": [], "mean_all": []}
        per_model_feats = {}
        for model_name in sorted(models_seen):
            key_real = (img_id, model_name, "REAL")
            key_fake = (img_id, model_name, "FAKE")
            if key_real in text_scores and key_fake in text_scores:
                real_s = text_scores[key_real]
                fake_s = text_scores[key_fake]
                for metric in ["caption", "mean_cue", "mean_all"]:
                    r_val = real_s[f"{metric}_score"]
                    f_val = fake_s[f"{metric}_score"]
                    disc = f_val - r_val
                    per_model_discs[metric].append(disc)
                    short = model_name.replace(".", "").replace("-", "")[:10]
                    per_model_feats[f"{short}_{metric}_disc"] = disc
                    per_model_feats[f"{short}_{metric}_real"] = r_val
                    per_model_feats[f"{short}_{metric}_fake"] = f_val
        if not per_model_discs["caption"]:
            continue
        feat = []
        for fn in feature_names:
            if fn in per_model_feats:
                feat.append(per_model_feats[fn])
            elif fn == "avg_caption_disc":
                feat.append(np.mean(per_model_discs["caption"]))
            elif fn == "avg_cue_disc":
                feat.append(np.mean(per_model_discs["mean_cue"]))
            elif fn == "avg_all_disc":
                feat.append(np.mean(per_model_discs["mean_all"]))
            elif fn == "std_caption_disc":
                feat.append(np.std(per_model_discs["caption"]) if len(per_model_discs["caption"]) > 1 else 0)
            elif fn == "std_cue_disc":
                feat.append(np.std(per_model_discs["mean_cue"]) if len(per_model_discs["mean_cue"]) > 1 else 0)
            elif fn == "std_all_disc":
                feat.append(np.std(per_model_discs["mean_all"]) if len(per_model_discs["mean_all"]) > 1 else 0)
            elif fn == "max_disc":
                feat.append(max(per_model_discs["mean_all"]))
            elif fn == "min_disc":
                feat.append(min(per_model_discs["mean_all"]))
            elif fn == "disc_range":
                feat.append(max(per_model_discs["mean_all"]) - min(per_model_discs["mean_all"]))
            elif fn == "avg_real_score":
                real_scores = [text_scores.get((img_id, m, "REAL"), {}).get("mean_all_score", 0)
                              for m in models_seen if (img_id, m, "REAL") in text_scores]
                feat.append(np.mean(real_scores) if real_scores else 0)
            elif fn == "avg_fake_score":
                fake_scores = [text_scores.get((img_id, m, "FAKE"), {}).get("mean_all_score", 0)
                              for m in models_seen if (img_id, m, "FAKE") in text_scores]
                feat.append(np.mean(fake_scores) if fake_scores else 0)
            elif fn == "n_models_agree_direction":
                feat.append(sum(1 for d in per_model_discs["mean_all"] if d > 0))
            else:
                feat.append(0.0)
        X.append(feat)
        y.append(label)
        ids.append(img_id)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"\nFeature matrix: {X.shape} (images x features)")
    print(f"Labels: {np.sum(y==0)} real, {np.sum(y==1)} fake")
    return X, y, ids, feature_names


def train_and_evaluate(X, y, ids, feature_names):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.preprocessing import StandardScaler

    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n" + "="*60)
    print("STEP 4: CLASSIFIER EVALUATION (5-Fold CV)")
    print("="*60)

    classifiers = {
        "GBM": GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=SEED),
        "LogReg": LogisticRegression(max_iter=1000, random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=SEED),
    }

    soft_labels = np.zeros(len(y), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for clf_name, clf in classifiers.items():
        fold_aucs, fold_accs = [], []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_te)[:, 1]
            preds = clf.predict(X_te)
            auc = roc_auc_score(y_te, probs)
            acc = accuracy_score(y_te, preds)
            fold_aucs.append(auc)
            fold_accs.append(acc)
            if clf_name == "GBM":
                soft_labels[test_idx] = probs
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        mean_acc = np.mean(fold_accs)
        print(f"\n{clf_name}:")
        print(f"  AUROC: {mean_auc:.4f} +/- {std_auc:.4f}")
        print(f"  Acc:   {mean_acc:.4f}")
        print(f"  Folds: {[f'{a:.3f}' for a in fold_aucs]}")

    print("\n-- Top 10 Features (GBM) --")
    clf_final = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=SEED)
    clf_final.fit(X_scaled, y)
    importances = clf_final.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(top_idx):
        print(f"  {rank+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    print("\n-- Baseline: Mean Discrepancy Only --")
    disc_idx = feature_names.index("avg_all_disc")
    disc_only = X[:, disc_idx]
    baseline_auc = roc_auc_score(y, disc_only)
    print(f"  AUROC (single feature): {baseline_auc:.4f}")

    return soft_labels, clf_final, scaler


def save_soft_labels(ids, y, soft_labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    records = []
    for img_id, gt, prob in zip(ids, y, soft_labels):
        records.append({
            "image_id": img_id,
            "ground_truth": "REAL" if gt == 0 else "FAKE",
            "teacher_prob_fake": float(prob),
            "teacher_prob_real": float(1 - prob),
        })
    out_path = os.path.join(output_dir, "teacher_soft_labels_3k.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSoft labels saved: {out_path}")
    return out_path


def main():
    t0 = time.time()
    print("="*60)
    print("DUAL-HYPOTHESIS PIPELINE - 3K IMAGE DEMO")
    print(f"Started: {datetime.now()}")
    print("="*60)

    print("\n-- STEP 0: Loading VLM results --")
    by_image, models_seen = load_all_vlm_results()
    if not by_image:
        print("ERROR: No VLM results found!")
        sys.exit(1)

    selected_ids = select_3k_subset(by_image)

    print("\n-- STEP 1: CLIP Feature Extraction --")
    image_features, text_scores = clip_extract(selected_ids, by_image)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clip_cache = os.path.join(OUTPUT_DIR, "clip_scores_3k.json")
    serializable = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in text_scores.items()}
    with open(clip_cache, "w") as f:
        json.dump(serializable, f)
    print(f"CLIP scores cached: {clip_cache}")

    print("\n-- STEPS 2-3: Discrepancy Features --")
    X, y, ids, feature_names = compute_features(selected_ids, by_image, text_scores, models_seen)

    soft_labels, clf, scaler = train_and_evaluate(X, y, ids, feature_names)

    print("\n-- STEP 5: Generating Teacher Soft Labels --")
    save_soft_labels(ids, y, soft_labels, OUTPUT_DIR)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/60:.1f} minutes")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
