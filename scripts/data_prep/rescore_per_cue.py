#!/usr/bin/env python3
"""
Per-cue CLIP rescoring with verbose progress output.
"""
import os, sys, json, time, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SRC        = "/NAS_DISK/Saptarshi_data/merged_5vlm_clipped.json"
DATASET    = "/NAS_DISK/Saptarshi_data/dataset"
OUT_SCORES = "/NAS_DISK/Saptarshi_data/per_cue_scores.json"
HF_CACHE   = "/NAS_DISK/Saptarshi_data/hf_cache"
DEVICE     = "cuda"
BATCH_IMG  = 128
BATCH_TXT  = 512

os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Unbuffered stdout so the tee log updates live
sys.stdout.reconfigure(line_buffering=True)

def log(msg, *, flush=True):
    print(msg, flush=flush)

def progress_bar(i, n, prefix="", suffix="", bar_len=30, start_time=None):
    frac = (i+1) / n
    filled = int(bar_len * frac)
    bar = "█" * filled + "░" * (bar_len - filled)
    eta_str = ""
    if start_time is not None and i > 0:
        elapsed = time.time() - start_time
        rate = (i+1) / elapsed
        eta = (n - i - 1) / rate if rate > 0 else 0
        eta_str = f" | {rate:.1f}/s | ETA {eta/60:5.1f}m"
    sys.stdout.write(f"\r  {prefix} [{bar}] {i+1:>6,}/{n:,} ({frac*100:5.1f}%){eta_str} {suffix}")
    sys.stdout.flush()
    if i + 1 == n:
        sys.stdout.write("\n")

log("="*70)
log("Per-cue CLIP rescoring — verbose mode")
log("="*70)

# ======================================================================
# [1/5] Load CLIP — show download progress explicitly
# ======================================================================
log("\n[1/5] Loading CLIP ViT-L/14...")
log("  NOTE: First run downloads ~1.7GB of weights. This is NOT a hang.")
log("  Watch ~/.cache/clip or the HF_CACHE for progress if unsure.")

t0 = time.time()
try:
    import open_clip
    log(f"  open_clip version: {open_clip.__version__ if hasattr(open_clip, '__version__') else 'unknown'}")
    log(f"  Requesting ViT-L-14 openai pretrained (with force_quick_gelu to match openai weights)...")

    # Fix the QuickGELU warning — openai weights were trained with QuickGELU
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
        cache_dir=HF_CACHE,
        force_quick_gelu=True,
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    USE_OPEN_CLIP = True
    log(f"  ✓ open_clip ViT-L-14 loaded in {time.time()-t0:.1f}s")
except Exception as e:
    log(f"  open_clip failed: {type(e).__name__}: {e}")
    log("  Falling back to transformers CLIP...")
    from transformers import CLIPModel, CLIPProcessor
    mn = "openai/clip-vit-large-patch14"
    log(f"  Downloading/loading {mn}...")
    model = CLIPModel.from_pretrained(mn, cache_dir=HF_CACHE)
    processor = CLIPProcessor.from_pretrained(mn, cache_dir=HF_CACHE)
    USE_OPEN_CLIP = False
    log(f"  ✓ transformers CLIP loaded in {time.time()-t0:.1f}s")

log(f"  Moving model to {DEVICE}...")
model = model.to(DEVICE).eval()
for p in model.parameters(): p.requires_grad_(False)
vram = torch.cuda.memory_allocated() / 1e9
log(f"  ✓ Model on GPU  |  VRAM used by model: {vram:.2f} GB")
free, total = torch.cuda.mem_get_info()
log(f"  ✓ VRAM free: {free/1e9:.1f} GB / {total/1e9:.1f} GB")

# ======================================================================
# [2/5] Load merged VLM results + find images
# ======================================================================
log(f"\n[2/5] Loading merged VLM results from {SRC}...")
t0 = time.time()
with open(SRC) as f: d = json.load(f)
results = d["results"]
log(f"  ✓ Loaded {len(results):,} entries in {time.time()-t0:.1f}s")
log(f"    Metadata: {d['metadata']}")

log("  Indexing image files on disk...")
real_dir = Path(DATASET) / "real/coco"
fake_dir = Path(DATASET) / "fake/midjourney"
IMG_EXT = (".jpg",".jpeg",".png",".webp")
real_stems = {p.stem: p for p in real_dir.iterdir() if p.suffix.lower() in IMG_EXT}
fake_stems = {p.stem: p for p in fake_dir.iterdir() if p.suffix.lower() in IMG_EXT}
all_stems = {**real_stems, **fake_stems}
log(f"    real/coco: {len(real_stems):,}")
log(f"    fake/midjourney: {len(fake_stems):,}")

unique_ids = sorted(set(r["image_id"] for r in results if r["image_id"] in all_stems))
missing = set(r["image_id"] for r in results) - set(all_stems.keys())
log(f"  ✓ Unique images to encode: {len(unique_ids):,}  |  missing on disk: {len(missing):,}")

# ======================================================================
# [3/5] Encode images
# ======================================================================
log(f"\n[3/5] Encoding {len(unique_ids):,} images (batch={BATCH_IMG})...")
img_embs = {}
t0 = time.time()

def preprocess_one(pth):
    img = Image.open(pth).convert("RGB")
    if USE_OPEN_CLIP:
        return preprocess(img)
    else:
        return processor(images=img, return_tensors="pt")["pixel_values"][0]

processed = 0
skipped = 0
for batch_start in range(0, len(unique_ids), BATCH_IMG):
    batch_ids = unique_ids[batch_start:batch_start+BATCH_IMG]
    imgs, good_ids = [], []
    for iid in batch_ids:
        try:
            imgs.append(preprocess_one(all_stems[iid]))
            good_ids.append(iid)
        except Exception as e:
            skipped += 1
    if imgs:
        x = torch.stack(imgs).to(DEVICE)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            if USE_OPEN_CLIP:
                emb = model.encode_image(x)
            else:
                emb = model.get_image_features(pixel_values=x)
        emb = F.normalize(emb.float(), dim=-1).cpu().numpy()
        for iid, e in zip(good_ids, emb):
            img_embs[iid] = e
    processed += len(batch_ids)
    progress_bar(processed-1, len(unique_ids), prefix="IMG",
                 suffix=f"cached={len(img_embs):,} skip={skipped}",
                 start_time=t0)

log(f"  ✓ {len(img_embs):,} image embeddings in {(time.time()-t0)/60:.1f} min")
log(f"    Skipped: {skipped}  |  Avg: {len(img_embs)/(time.time()-t0):.1f} img/s")

# ======================================================================
# [4/5] Score caption + 3 cues for every VLM entry
# ======================================================================
log(f"\n[4/5] Scoring captions + cues (batch={BATCH_TXT})...")
log(f"      Total text encodings: {len(results)*4:,}")

def tokenize(texts):
    if USE_OPEN_CLIP:
        return tokenizer(texts).to(DEVICE)
    else:
        return processor(text=texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=77).to(DEVICE)

def encode_text_batch(texts):
    toks = tokenize(texts)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        if USE_OPEN_CLIP:
            t_emb = model.encode_text(toks)
        else:
            t_emb = model.get_text_features(**toks)
    return F.normalize(t_emb.float(), dim=-1)

per_entry = [None] * len(results)
buf_texts, buf_meta = [], []
t0 = time.time()
text_ops = 0
TOTAL_TEXT = len(results) * 4

def flush_text_batch():
    global text_ops
    if not buf_texts: return
    t_emb = encode_text_batch(buf_texts).cpu().numpy()
    for (idx, field), te in zip(buf_meta, t_emb):
        iid = results[idx]["image_id"]
        if iid not in img_embs: continue
        sim = float(np.dot(img_embs[iid], te))
        if per_entry[idx] is None:
            per_entry[idx] = {
                "image_id": iid,
                "model":    results[idx]["model"],
                "assumption": results[idx]["assumption"],
                "ground_truth": results[idx]["ground_truth"],
            }
        per_entry[idx][f"clip_{field}"] = sim
    text_ops += len(buf_texts)

entry_skipped = 0
for idx, r in enumerate(results):
    if r["image_id"] not in img_embs:
        entry_skipped += 1; continue
    for field in ("caption", "cue_1", "cue_2", "cue_3"):
        text = (r.get(field) or "").strip()
        if not text:
            if per_entry[idx] is None:
                per_entry[idx] = {"image_id": r["image_id"], "model": r["model"],
                                  "assumption": r["assumption"], "ground_truth": r["ground_truth"]}
            per_entry[idx][f"clip_{field}"] = None
            continue
        buf_texts.append(text[:300])
        buf_meta.append((idx, field))
        if len(buf_texts) >= BATCH_TXT:
            flush_text_batch()
            buf_texts, buf_meta = [], []
            progress_bar(text_ops-1, TOTAL_TEXT, prefix="TXT",
                         suffix=f"entries_done={idx:,}", start_time=t0)
flush_text_batch()
progress_bar(TOTAL_TEXT-1, TOTAL_TEXT, prefix="TXT",
             suffix=f"done", start_time=t0)

scored = sum(1 for e in per_entry if e is not None)
log(f"  ✓ {scored:,} entries scored in {(time.time()-t0)/60:.1f} min")
log(f"    Skipped (image missing): {entry_skipped:,}")

log(f"\n  Writing {OUT_SCORES}...")
out = [e for e in per_entry if e is not None]
with open(OUT_SCORES, "w") as f:
    json.dump({"metadata": {"total": len(out), "clip_model": "ViT-L-14",
                            "fields": ["clip_caption","clip_cue_1","clip_cue_2","clip_cue_3"]},
               "results": out}, f)
log(f"  ✓ Wrote {Path(OUT_SCORES).stat().st_size/1e6:.1f} MB")

# ======================================================================
# [5/5] Build teacher features + eval variants
# ======================================================================
log("\n[5/5] Building teacher features and evaluating...")
by_img = defaultdict(lambda: defaultdict(dict))
gt_map = {}
for e in out:
    by_img[e["image_id"]][e["model"]][e["assumption"]] = {
        "cap": e.get("clip_caption"),
        "c1":  e.get("clip_cue_1"),
        "c2":  e.get("clip_cue_2"),
        "c3":  e.get("clip_cue_3"),
    }
    gt_map[e["image_id"]] = e["ground_truth"]
log(f"  ✓ Indexed {len(by_img):,} images into feature builder")

FIELDS = ["cap", "c1", "c2", "c3"]

def per_image_features(vlm_dict, exclude=set()):
    reals = {f: [] for f in FIELDS}
    fakes = {f: [] for f in FIELDS}
    diffs = {f: [] for f in FIELDS}
    for vlm, asm in vlm_dict.items():
        if vlm in exclude: continue
        if "REAL" not in asm or "FAKE" not in asm: continue
        for f in FIELDS:
            r, fk = asm["REAL"].get(f), asm["FAKE"].get(f)
            if r is None or fk is None: continue
            reals[f].append(r); fakes[f].append(fk); diffs[f].append(fk - r)
    feat = []
    for f in FIELDS: feat.append(np.mean(reals[f]) if reals[f] else 0.0)
    for f in FIELDS: feat.append(np.mean(fakes[f]) if fakes[f] else 0.0)
    for f in FIELDS: feat.append(np.mean(diffs[f]) if diffs[f] else 0.0)
    for f in FIELDS: feat.append(np.std(diffs[f]) if len(diffs[f])>1 else 0.0)
    for f in FIELDS: feat.append(np.mean([d>0 for d in diffs[f]]) if diffs[f] else 0.5)
    return feat

def eval_variant(exclude, tag):
    log(f"\n  === Variant: {tag}  (excluding: {exclude or 'nothing'}) ===")
    ids, feats, labels = [], [], []
    for iid, vlm_dict in by_img.items():
        f = per_image_features(vlm_dict, exclude=exclude)
        if sum(abs(x) for x in f) == 0: continue
        ids.append(iid); feats.append(f); labels.append(1 if gt_map[iid]=="FAKE" else 0)
    feats = np.array(feats, dtype=np.float32); labels = np.array(labels)
    log(f"    n = {len(ids):,}  |  REAL={int((labels==0).sum()):,} FAKE={int((labels==1).sum()):,}")

    feat_names = ([f"mean_real_{f}" for f in FIELDS] + [f"mean_fake_{f}" for f in FIELDS] +
                  [f"diff_{f}" for f in FIELDS] + [f"diffstd_{f}" for f in FIELDS] +
                  [f"votes_{f}" for f in FIELDS])

    log(f"    ── Per-field discrepancy AUROC (THE KEY COMPARISON) ──")
    for i, f in enumerate(FIELDS):
        col = 8 + i
        a = roc_auc_score(labels, feats[:, col])
        gap = feats[labels==1,col].mean() - feats[labels==0,col].mean()
        log(f"      diff_{f:3s}: AUROC={a:.4f}  gap={gap:+.5f}  "
            f"(REAL={feats[labels==0,col].mean():+.5f}, FAKE={feats[labels==1,col].mean():+.5f})")

    best_feat_auc, best_feat_idx = 0, -1
    for i in range(feats.shape[1]):
        try:
            a = roc_auc_score(labels, feats[:, i])
            if max(a, 1-a) > best_feat_auc:
                best_feat_auc = max(a, 1-a); best_feat_idx = i
        except: pass
    log(f"    Best single feature overall: {feat_names[best_feat_idx]} = {best_feat_auc:.4f}")

    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    clf.fit(feats, labels)
    p = clf.predict_proba(feats)[:, 1]
    auroc_full = roc_auc_score(labels, p)
    log(f"    Full 20-feat logistic AUROC: {auroc_full:.4f}")

    probs = np.stack([1-p, p], axis=1).astype(np.float32)
    suffix = f"_{tag}" if tag != "all" else ""
    path = f"/NAS_DISK/Saptarshi_data/teacher_soft_labels_percue{suffix}.npz"
    np.savez_compressed(path, ids=np.array(ids), probs=probs,
                        feats=feats, labels=labels,
                        logreg_coef=clf.coef_, logreg_intercept=clf.intercept_)
    log(f"    ✓ Saved -> {path}")
    return auroc_full

log("\n" + "="*70)
log("HEADLINE RESULTS — does per-cue scoring fix the teacher?")
log("="*70)
auc_all = eval_variant(exclude=set(), tag="all")
auc_nop = eval_variant(exclude={"Phi-4-multimodal"}, tag="no_phi4")
auc_glm = eval_variant(
    exclude={"InternVL2.5-8B-MPO","Pixtral-12B","Phi-4-multimodal","Qwen2.5-VL-7B-Instruct"},
    tag="glm4v_only")

log("\n" + "="*70)
log("COMPARISON TABLE")
log("="*70)
log(f"  OLD teacher (caption+cues blob):      0.7002")
log(f"  Direct CLIP baseline (no VLMs):       0.8715")
log(f"  NEW teacher (per-cue), all 5 VLMs:   {auc_all:.4f}")
log(f"  NEW teacher (per-cue), no Phi-4:     {auc_nop:.4f}")
log(f"  NEW teacher (per-cue), GLM-4V only:  {auc_glm:.4f}")
best = max(auc_all, auc_nop, auc_glm)
delta = best - 0.7002
log(f"\n  Best per-cue variant:                {best:.4f}  (Δ vs old = {delta:+.4f})")
if best > 0.85:
    log(f"  ✓✓ Method SALVAGED. Continue with distillation.")
elif best > 0.78:
    log(f"  ✓  Solid improvement. Method usable but not heroic.")
elif delta > 0.05:
    log(f"  ~  Modest gain. Thesis needs reframing.")
else:
    log(f"  ✗  No meaningful improvement. Method fundamentally limited.")
