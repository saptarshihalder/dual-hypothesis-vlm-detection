import json, os, time, torch, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict, Counter
import open_clip
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
import pandas as pd

BASE = Path("/NAS_DISK/Saptarshi_data")
REAL_DIR = BASE / "dataset" / "real" / "coco"
MJ_DIR   = BASE / "dataset" / "fake" / "midjourney"
GAN_BASE = BASE / "dataset" / "fake" / "gan_test"
DEV = "cpu"

print("="*60)
print("  CROSS-GENERATOR EVALUATION")
print("  Train: Midjourney | Test: starGAN, BigGAN, styleGAN")
print("="*60)

print("\n[1/4] Loading CLIP ViT-L-14...")
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=DEV)
tokenizer = open_clip.get_tokenizer("ViT-L-14")
clip_model.eval()
print("  CLIP loaded")

def find_image(image_id, ground_truth, generator="midjourney"):
    if ground_truth == "REAL":
        base = REAL_DIR
    elif generator in ("starGAN", "BigGAN", "styleGAN"):
        base = GAN_BASE / generator
    else:
        base = MJ_DIR
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPEG", ".PNG"]:
        p = base / f"{image_id}{ext}"
        if p.exists():
            return str(p)
    matches = list(base.glob(f"{image_id}.*"))
    return str(matches[0]) if matches else None

img_cache = {}

def get_img_feat(image_id, ground_truth, generator="midjourney"):
    if image_id in img_cache:
        return img_cache[image_id]
    path = find_image(image_id, ground_truth, generator)
    if path is None:
        img_cache[image_id] = None
        return None
    try:
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEV)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            feat = clip_model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        img_cache[image_id] = feat
        return feat
    except:
        img_cache[image_id] = None
        return None

def get_txt_feat(text):
    tokens = tokenizer([text]).to(DEV)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        feat = clip_model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat

def clip_score_results(results):
    scored, errors = 0, 0
    for r in tqdm(results, desc="  CLIP scoring", unit="entry"):
        parts = [r.get("caption", "")]
        for k in ["cue_1", "cue_2", "cue_3"]:
            if r.get(k):
                parts.append(r[k])
        text = ". ".join(p for p in parts if p)
        gen = r.get("generator", "midjourney")
        img_feat = get_img_feat(r["image_id"], r["ground_truth"], gen)
        if img_feat is None:
            r["clip_score"] = None
            errors += 1
            continue
        try:
            txt_feat = get_txt_feat(text)
            r["clip_score"] = round(float((img_feat @ txt_feat.T).item()), 6)
            scored += 1
        except:
            r["clip_score"] = None
            errors += 1
        if len(img_cache) > 3000:
            img_cache.clear()
            torch.cuda.empty_cache()
    print(f"  Scored: {scored}, Errors: {errors}")
    return results

def build_features(results):
    images = defaultdict(lambda: {})
    for r in results:
        if r.get("clip_score") is None:
            continue
        img_id = r["image_id"]
        images[img_id]["ground_truth"] = r["ground_truth"]
        images[img_id]["generator"] = r.get("generator", "unknown")
        images[img_id][r["assumption"]] = r["clip_score"]
    rows, labels, gens = [], [], []
    for img_id, info in images.items():
        if "REAL" not in info or "FAKE" not in info:
            continue
        disc = info["FAKE"] - info["REAL"]
        rows.append({"disc": disc, "real_score": info["REAL"], "fake_score": info["FAKE"]})
        labels.append(1 if info["ground_truth"] == "FAKE" else 0)
        gens.append(info.get("generator", "unknown"))
    return rows, np.array(labels), gens

print("\n[2/4] Loading Midjourney Qwen data (already CLIP-scored)...")
with open(BASE / "merged_5vlm_clipped.json") as f:
    mj_data = json.load(f)
mj_qwen = [r for r in mj_data["results"] if "Qwen" in r["model"]]
print(f"  Midjourney Qwen entries: {len(mj_qwen)}, with scores: {sum(1 for r in mj_qwen if r.get('clip_score') is not None)}")

print("\n[3/4] CLIP-scoring GAN test data...")
with open(BASE / "gan_vlm_results" / "gan_Qwen.json") as f:
    gan_data = json.load(f)
gan_results = clip_score_results(gan_data["results"])
img_cache.clear()
torch.cuda.empty_cache()

print("\n[4/4] Cross-generator evaluation...")
mj_rows, mj_labels, mj_gens = build_features(mj_qwen)
gan_rows, gan_labels, gan_gens = build_features(gan_results)
print(f"  Train (Midjourney): {len(mj_rows)} images, REAL={sum(mj_labels==0)}, FAKE={sum(mj_labels==1)}")
print(f"  Test (GAN):         {len(gan_rows)} images, REAL={sum(gan_labels==0)}, FAKE={sum(gan_labels==1)}")

X_train = pd.DataFrame(mj_rows).fillna(0).values
y_train = mj_labels
X_test = pd.DataFrame(gan_rows).fillna(0).values
y_test = gan_labels

print("\n" + "="*60)
print("  RESULTS: Train Midjourney -> Test GANs (Qwen single-VLM)")
print("="*60)

for name, clf in [
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("Gradient Boosting", GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42))]:
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    print(f"\n  {name}:")
    print(f"    Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"    Overall AUROC:    {roc_auc_score(y_test, y_prob):.4f}")
    print(f"    Overall F1:       {f1_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["REAL","FAKE"]))
    print("  Per-Generator Results:")
    for gen in sorted(set(g for g, l in zip(gan_gens, gan_labels) if l == 1)):
        mask = np.array([(g == gen and l == 1) or (l == 0) for g, l in zip(gan_gens, gan_labels)])
        if mask.sum() < 10:
            continue
        y_sub = y_test[mask]
        prob_sub = y_prob[mask]
        pred_sub = y_pred[mask]
        try:
            auc = roc_auc_score(y_sub, prob_sub)
            acc = accuracy_score(y_sub, pred_sub)
            print(f"    {gen:12s}  AUROC={auc:.4f}  Acc={acc:.4f}  (n={sum(y_sub==1)})")
        except:
            print(f"    {gen:12s}  Could not compute")

print("\n  In-Distribution (Midjourney) - 5-fold CV:")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_res = cross_validate(
    Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
    X_train, y_train, cv=cv, scoring=["accuracy", "roc_auc"])
print(f"    Midjourney Acc:   {cv_res['test_accuracy'].mean():.4f}")
print(f"    Midjourney AUROC: {cv_res['test_roc_auc'].mean():.4f}")
print("\nDONE!")
