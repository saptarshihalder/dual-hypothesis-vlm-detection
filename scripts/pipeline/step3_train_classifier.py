#!/usr/bin/env python3
"""Step 3: Compute discrepancy features and train classifier.

For each image:
  - Per VLM: discrepancy = clip_score(FAKE assumption) - clip_score(REAL assumption)
  - Aggregate across VLMs: mean, std, min, max of discrepancies
  - Also individual VLM discrepancies as features

Then train classifiers (Logistic Regression, SVM, Random Forest, XGBoost)
with stratified 5-fold cross-validation.
"""

import json, os, sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# Try importing xgboost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed, skipping. Install with: pip install xgboost")

# ── CONFIG ──
INPUT = "/NAS_DISK/Saptarshi_data/merged_5vlm_clipped.json"
OUTPUT_DIR = "/NAS_DISK/Saptarshi_data/classifier_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD ──
print("Loading CLIP-scored VLM results...")
with open(INPUT) as f:
    data = json.load(f)
results = data["results"]
print(f"Total entries: {len(results)}")

# ── ORGANIZE BY IMAGE ──
# Structure: images[image_id] = {
#   "ground_truth": ...,
#   "scores": { model: { "REAL": clip_score, "FAKE": clip_score } }
# }
images = defaultdict(lambda: {"scores": {}})
skipped = 0

for r in results:
    img_id = r["image_id"]
    model = r["model"]
    assume = r["assumption"]
    score = r.get("clip_score")

    if score is None:
        skipped += 1
        continue

    images[img_id]["ground_truth"] = r["ground_truth"]
    if model not in images[img_id]["scores"]:
        images[img_id]["scores"][model] = {}
    images[img_id]["scores"][model][assume] = score

print(f"Unique images with scores: {len(images)}")
print(f"Skipped (no CLIP score): {skipped}")

# ── GET MODEL LIST ──
all_models = sorted(set(r["model"] for r in results))
print(f"Models: {all_models}")

# ── BUILD FEATURE MATRIX ──
print("\nBuilding feature matrix...")
feature_rows = []
labels = []
image_ids = []

for img_id, info in images.items():
    gt = info["ground_truth"]
    model_scores = info["scores"]

    # Need both REAL and FAKE scores for at least 3 models
    discrepancies = []
    row = {}

    for model in all_models:
        if model in model_scores:
            real_s = model_scores[model].get("REAL")
            fake_s = model_scores[model].get("FAKE")
            if real_s is not None and fake_s is not None:
                disc = fake_s - real_s
                discrepancies.append(disc)
                # Short model name for column
                short = model.split("-")[0].split("/")[-1]
                row[f"disc_{short}"] = disc
                row[f"real_{short}"] = real_s
                row[f"fake_{short}"] = fake_s

    if len(discrepancies) < 3:
        continue  # skip if too few models succeeded

    # Aggregate features
    row["disc_mean"] = np.mean(discrepancies)
    row["disc_std"] = np.std(discrepancies)
    row["disc_min"] = np.min(discrepancies)
    row["disc_max"] = np.max(discrepancies)
    row["disc_range"] = np.max(discrepancies) - np.min(discrepancies)
    row["disc_median"] = np.median(discrepancies)
    row["num_positive_disc"] = sum(1 for d in discrepancies if d > 0)
    row["num_models"] = len(discrepancies)

    feature_rows.append(row)
    labels.append(gt)
    image_ids.append(img_id)

df = pd.DataFrame(feature_rows)
y = np.array([1 if l == "FAKE" else 0 for l in labels])

print(f"Feature matrix: {df.shape}")
print(f"Class distribution: REAL={sum(y==0)}, FAKE={sum(y==1)}")
print(f"Features: {list(df.columns)}")

# ── HANDLE MISSING ──
df = df.fillna(0)

# ── TRAIN CLASSIFIERS ──
print("\n" + "="*60)
print("TRAINING CLASSIFIERS (5-fold stratified CV)")
print("="*60)

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, C=1.0),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
}
if HAS_XGB:
    classifiers["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_summary = {}

for name, clf in classifiers.items():
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

    scoring = ["accuracy", "roc_auc", "f1"]
    cv_results = cross_validate(
        pipe, df.values, y, cv=cv, scoring=scoring, return_train_score=True
    )

    acc = cv_results["test_accuracy"]
    auc = cv_results["test_roc_auc"]
    f1 = cv_results["test_f1"]

    print(f"  Accuracy:  {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"  AUROC:     {auc.mean():.4f} ± {auc.std():.4f}")
    print(f"  F1:        {f1.mean():.4f} ± {f1.std():.4f}")

    results_summary[name] = {
        "accuracy": f"{acc.mean():.4f} ± {acc.std():.4f}",
        "auroc": f"{auc.mean():.4f} ± {auc.std():.4f}",
        "f1": f"{f1.mean():.4f} ± {f1.std():.4f}",
        "accuracy_mean": float(acc.mean()),
        "auroc_mean": float(auc.mean()),
    }

# ── BEST MODEL: Full train + detailed report ──
print("\n" + "="*60)
print("DETAILED REPORT (best model on full data)")
print("="*60)

best_name = max(results_summary, key=lambda k: results_summary[k]["auroc_mean"])
print(f"Best model: {best_name}")

best_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", classifiers[best_name]),
])

# 80/20 split for final report
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    df.values, y, image_ids, test_size=0.2, stratify=y, random_state=42
)

best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)
y_prob = best_clf.predict_proba(X_test)[:, 1]

print(f"\nTest set results ({len(y_test)} images):")
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"  AUROC:    {roc_auc_score(y_test, y_prob):.4f}")
print(f"  F1:       {f1_score(y_test, y_pred):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  REAL predicted REAL: {cm[0][0]}, predicted FAKE: {cm[0][1]}")
print(f"  FAKE predicted REAL: {cm[1][0]}, predicted FAKE: {cm[1][1]}")

# ── FEATURE IMPORTANCE ──
if hasattr(classifiers[best_name], "feature_importances_"):
    importances = best_clf.named_steps["clf"].feature_importances_
    feat_imp = sorted(zip(df.columns, importances), key=lambda x: -x[1])
    print(f"\nFeature Importance ({best_name}):")
    for fname, imp in feat_imp:
        print(f"  {fname:25s} {imp:.4f}")

# ── ABLATION: Single VLM vs Multi-VLM ──
print("\n" + "="*60)
print("ABLATION: Single VLM vs Multi-VLM")
print("="*60)

for model in all_models:
    short = model.split("-")[0].split("/")[-1]
    single_cols = [c for c in df.columns if short in c]
    if not single_cols:
        continue
    X_single = df[single_cols].fillna(0).values
    pipe_single = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    cv_res = cross_validate(pipe_single, X_single, y, cv=cv,
                            scoring=["accuracy", "roc_auc"])
    acc = cv_res["test_accuracy"].mean()
    auc = cv_res["test_roc_auc"].mean()
    print(f"  {model:30s}  Acc={acc:.4f}  AUROC={auc:.4f}")

# Multi-VLM (all features)
pipe_all = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])
cv_res = cross_validate(pipe_all, df.values, y, cv=cv,
                        scoring=["accuracy", "roc_auc"])
print(f"  {'ALL 5 VLMs (combined)':30s}  Acc={cv_res['test_accuracy'].mean():.4f}  "
      f"AUROC={cv_res['test_roc_auc'].mean():.4f}")

# ── ABLATION: With vs Without Dual Hypothesis ──
print("\n" + "="*60)
print("ABLATION: With vs Without Dual Hypothesis")
print("="*60)

# Without dual: use only REAL-assumption scores
real_only_cols = [c for c in df.columns if c.startswith("real_")]
fake_only_cols = [c for c in df.columns if c.startswith("fake_")]
disc_cols = [c for c in df.columns if c.startswith("disc_") or c.startswith("num_")]

for label, cols in [("REAL scores only", real_only_cols),
                    ("FAKE scores only", fake_only_cols),
                    ("Discrepancy features (dual)", disc_cols),
                    ("ALL features", list(df.columns))]:
    if not cols:
        continue
    X_sub = df[cols].fillna(0).values
    pipe_sub = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    cv_res = cross_validate(pipe_sub, X_sub, y, cv=cv,
                            scoring=["accuracy", "roc_auc"])
    print(f"  {label:35s}  Acc={cv_res['test_accuracy'].mean():.4f}  "
          f"AUROC={cv_res['test_roc_auc'].mean():.4f}")

# ── SAVE RESULTS ──
summary = {
    "dataset": {
        "total_images": len(image_ids),
        "real": int(sum(y == 0)),
        "fake": int(sum(y == 1)),
        "features": list(df.columns),
        "num_features": len(df.columns),
    },
    "cv_results": results_summary,
    "best_model": best_name,
}

with open(f"{OUTPUT_DIR}/classifier_results.json", "w") as f:
    json.dump(summary, f, indent=2)

# Save feature matrix for further analysis
df["ground_truth"] = labels
df["image_id"] = image_ids
df.to_csv(f"{OUTPUT_DIR}/feature_matrix.csv", index=False)

print(f"\nResults saved to {OUTPUT_DIR}/")
print("Done!")
