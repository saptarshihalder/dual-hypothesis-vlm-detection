#!/usr/bin/env python3
"""
Train CNNSpot, NPR, and UniFD baselines on Midjourney+COCO (same split as DHSDv2),
then evaluate on the 13-generator CNNDetection cross-gen test split.

Saves per-image predictions to .npz files matching DHSDv2's prediction format
so all methods can be compared head-to-head.

Usage:
    python3 train_baselines.py
    python3 train_baselines.py --method cnnspot
    python3 train_baselines.py --method npr
    python3 train_baselines.py --method univfd
    python3 train_baselines.py --eval-only
"""
import os, sys, json, argparse, random, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageFilter
import io
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

DATA_ROOT = Path("/NAS_DISK/Saptarshi_data/dataset")
MJ_DIR    = DATA_ROOT / "fake" / "midjourney"
COCO_DIR  = DATA_ROOT / "real" / "coco"
CNN_ROOT  = DATA_ROOT / "cnndetection_test"
TPROB_NPZ = Path("/NAS_DISK/Saptarshi_data/hybrid_teacher_soft_labels.npz")

OUT_DIR   = Path("/NAS_DISK/Saptarshi_data/baselines_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
IMG_SIZE    = 224
BATCH_TRAIN = 32
BATCH_EVAL  = 64
NUM_WORK    = 6
EPOCHS      = 5
LR          = 1e-4
WD          = 0.01
DEVICE      = "cuda"
IMG_EXT     = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
CROSSGEN_VAL_PER_GEN = 250

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.benchmark = True


def build_train_items():
    if not TPROB_NPZ.exists():
        raise FileNotFoundError(f"{TPROB_NPZ} required to match DHSDv2 split")
    teacher_ids = set(np.load(TPROB_NPZ, allow_pickle=True)["ids"].tolist())
    items = []
    for d, lbl in [(MJ_DIR, 1), (COCO_DIR, 0)]:
        for p in sorted(d.glob("*")):
            if p.suffix.lower() in IMG_EXT and p.stem in teacher_ids:
                items.append((str(p), lbl))
    rng = random.Random(SEED)
    rng.shuffle(items)
    n = len(items)
    n_test = int(n * 0.15)
    n_val  = int(n * 0.15)
    n_tr   = n - n_test - n_val
    return items[:n_tr], items[n_tr:n_tr+n_val], items[n_tr+n_val:]


def build_crossgen_split():
    gen_items = defaultdict(list)
    for gen_dir in sorted(CNN_ROOT.iterdir()):
        if not gen_dir.is_dir(): continue
        for sub, lbl in [("0_real", 0), ("1_fake", 1)]:
            d = gen_dir / sub
            if d.exists():
                for p in d.iterdir():
                    if p.suffix.lower() in IMG_EXT:
                        gen_items[gen_dir.name].append((str(p), lbl))
        for sub in gen_dir.iterdir():
            if not sub.is_dir() or sub.name in ("0_real", "1_fake"): continue
            for inner, ilbl in [("0_real", 0), ("1_fake", 1)]:
                d = sub / inner
                if d.exists():
                    for p in d.iterdir():
                        if p.suffix.lower() in IMG_EXT:
                            gen_items[gen_dir.name].append((str(p), ilbl))

    rng = random.Random(SEED + 100)
    val, test = defaultdict(list), defaultdict(list)
    for gen, items in sorted(gen_items.items()):
        if set(l for _, l in items) != {0, 1}: continue
        real = [x for x in items if x[1] == 0]
        fake = [x for x in items if x[1] == 1]
        rng.shuffle(real); rng.shuffle(fake)
        nv = min(CROSSGEN_VAL_PER_GEN, len(real)//2, len(fake)//2)
        val[gen]  = real[:nv] + fake[:nv]
        test[gen] = real[nv:] + fake[nv:]
    return val, test


def random_jpeg(img, q_lo=60, q_hi=100):
    q = random.randint(q_lo, q_hi)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def random_blur(img, sigma_lo=0.0, sigma_hi=3.0):
    s = random.uniform(sigma_lo, sigma_hi)
    if s < 0.05: return img
    return img.filter(ImageFilter.GaussianBlur(radius=s))


class CNNSpotAug:
    def __init__(self, train):
        self.train = train
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, img):
        img = img.convert("RGB")
        if self.train:
            if random.random() < 0.5: img = random_blur(img)
            if random.random() < 0.5: img = random_jpeg(img)
            img = transforms.functional.resize(img, IMG_SIZE)
            img = transforms.RandomCrop(IMG_SIZE, pad_if_needed=True)(img)
            if random.random() < 0.5: img = transforms.functional.hflip(img)
        else:
            img = transforms.functional.resize(img, IMG_SIZE)
            img = transforms.functional.center_crop(img, IMG_SIZE)
        img = transforms.functional.to_tensor(img)
        return self.norm(img)


class NPRAug:
    def __init__(self, train):
        self.train = train
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, img):
        img = img.convert("RGB")
        img = transforms.functional.resize(img, IMG_SIZE)
        if self.train:
            img = transforms.RandomCrop(IMG_SIZE, pad_if_needed=True)(img)
            if random.random() < 0.5: img = transforms.functional.hflip(img)
        else:
            img = transforms.functional.center_crop(img, IMG_SIZE)
        img = transforms.functional.to_tensor(img)
        return self.norm(img)


class UniFDAug:
    """CLIP standard preprocessing for UniFD (Ojha et al. 2023)."""
    def __init__(self, train):
        self.train = train
        self.norm = transforms.Normalize(
            mean=[0.4815, 0.4578, 0.4082],
            std=[0.2686, 0.2613, 0.2758])

    def __call__(self, img):
        img = img.convert("RGB")
        img = transforms.functional.resize(img, IMG_SIZE,
            interpolation=transforms.InterpolationMode.BICUBIC)
        if self.train:
            img = transforms.RandomCrop(IMG_SIZE, pad_if_needed=True)(img)
            if random.random() < 0.5: img = transforms.functional.hflip(img)
        else:
            img = transforms.functional.center_crop(img, IMG_SIZE)
        img = transforms.functional.to_tensor(img)
        return self.norm(img)


class ImgListDS(Dataset):
    def __init__(self, items, transform):
        self.items, self.tfm = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, lbl = self.items[i]
        try:
            x = self.tfm(Image.open(path))
        except Exception:
            x = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        return x, int(lbl)


def make_resnet50_binary():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


class NPRWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = make_resnet50_binary()

    def forward(self, x):
        n, c, h, w = x.shape
        if h % 2 == 1: x = x[:, :, :-1, :]
        if w % 2 == 1: x = x[:, :, :, :-1]
        down = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        up   = F.interpolate(down, scale_factor=2.0, mode="bilinear", align_corners=False)
        npr  = x - up
        return self.backbone(npr)


class UniFDProbe(nn.Module):
    """Ojha et al. 2023 (UniversalFakeDetect): frozen CLIP-ViT-L/14 image
    encoder + a single linear classifier. CLIP-based baseline isolating
    the contribution of dual-hypothesis reasoning over vanilla CLIP features."""
    def __init__(self):
        super().__init__()
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai",
            cache_dir="/NAS_DISK/Saptarshi_data/hf_cache")
        self.visual = clip_model.visual
        for p in self.visual.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        with torch.no_grad():
            feat = self.visual(x)
        return self.fc(feat.float())


def get_model_and_transforms(method):
    if method == "cnnspot":
        return make_resnet50_binary(), CNNSpotAug(True), CNNSpotAug(False)
    elif method == "npr":
        return NPRWrapper(), NPRAug(True), NPRAug(False)
    elif method == "univfd":
        return UniFDProbe(), UniFDAug(True), UniFDAug(False)
    else:
        raise ValueError(method)


def train_one(method, train_items, val_items):
    print(f"\n{'='*70}\n  Training {method.upper()}\n{'='*70}")

    model, TrainT, EvalT = get_model_and_transforms(method)
    model = model.to(DEVICE)

    train_ld = DataLoader(ImgListDS(train_items, TrainT),
                          batch_size=BATCH_TRAIN, shuffle=True,
                          num_workers=NUM_WORK, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(ImgListDS(val_items, EvalT),
                          batch_size=BATCH_EVAL, shuffle=False,
                          num_workers=NUM_WORK, pin_memory=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    opt   = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    bce   = nn.BCEWithLogitsLoss()

    best_val_auroc = 0.0
    best_path = OUT_DIR / f"{method}_best.pt"
    log = []

    for ep in range(EPOCHS):
        model.train()
        t0 = time.time(); losses = []
        for x, y in train_ld:
            x = x.to(DEVICE, non_blocking=True)
            y = y.float().to(DEVICE, non_blocking=True)
            opt.zero_grad()
            logit = model(x).squeeze(-1)
            loss  = bce(logit, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in val_ld:
                x = x.to(DEVICE, non_blocking=True)
                p = torch.sigmoid(model(x).squeeze(-1)).cpu().numpy()
                ps.extend(p); ys.extend(y.numpy())
        ys, ps = np.array(ys), np.array(ps)
        val_auroc = roc_auc_score(ys, ps)
        val_acc   = accuracy_score(ys, (ps > 0.5).astype(int))
        elapsed = time.time() - t0

        log.append({"epoch": ep, "train_loss": float(np.mean(losses)),
                    "val_auroc": val_auroc, "val_acc": val_acc, "sec": elapsed})
        print(f"  ep{ep}  loss={np.mean(losses):.4f}  "
              f"val_AUROC={val_auroc:.4f}  val_acc={val_acc:.4f}  "
              f"({elapsed:.0f}s)")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save({"state": model.state_dict(),
                        "epoch": ep, "val_auroc": val_auroc,
                        "method": method}, best_path)
            print(f"      saved {best_path.name}")

    with open(OUT_DIR / f"{method}_train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Best {method} val AUROC: {best_val_auroc:.4f}")
    return best_path


def evaluate(method, ckpt_path, crossgen_test):
    print(f"\n{'='*70}\n  Evaluating {method.upper()} on cross-gen test\n{'='*70}")

    model, _, EvalT = get_model_and_transforms(method)
    model = model.to(DEVICE)

    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state"]); model.eval()
    print(f"  Loaded {ckpt_path.name} (val AUROC {ck['val_auroc']:.4f})")

    results = {}
    pred_blob = {"generators": np.array(sorted(crossgen_test.keys()))}

    for gen in sorted(crossgen_test.keys()):
        items = crossgen_test[gen]
        if not items: continue
        ld = DataLoader(ImgListDS(items, EvalT), batch_size=BATCH_EVAL,
                        shuffle=False, num_workers=NUM_WORK, pin_memory=True)
        ys, ps = [], []
        with torch.no_grad():
            for x, y in ld:
                x = x.to(DEVICE, non_blocking=True)
                p = torch.sigmoid(model(x).squeeze(-1)).cpu().numpy()
                ps.extend(p); ys.extend(y.numpy())
        ys, ps = np.array(ys), np.array(ps)
        pred_blob[f"{gen}_labels"] = ys
        pred_blob[f"{gen}_probs"]  = ps

        auroc = roc_auc_score(ys, ps)
        ap    = average_precision_score(ys, ps)
        acc   = accuracy_score(ys, (ps > 0.5).astype(int))
        rmask, fmask = ys == 0, ys == 1
        macc = (accuracy_score(ys[rmask], (ps[rmask] > 0.5).astype(int)) +
                accuracy_score(ys[fmask], (ps[fmask] > 0.5).astype(int))) / 2
        results[gen] = dict(n=len(ys), auroc=float(auroc), ap=float(ap),
                            acc=float(acc), mAcc=float(macc))
        print(f"  {gen:<22} n={len(ys):>5d}  AUROC={auroc:.4f}  "
              f"AP={ap:.4f}  mAcc={macc:.4f}")

    np.savez(OUT_DIR / f"{method}_crossgen_predictions.npz", **pred_blob)

    macro = dict(
        macro_auroc=float(np.mean([r["auroc"] for r in results.values()])),
        macro_ap=float(np.mean([r["ap"] for r in results.values()])),
        macro_acc=float(np.mean([r["acc"] for r in results.values()])),
        macro_mAcc=float(np.mean([r["mAcc"] for r in results.values()])),
    )
    with open(OUT_DIR / f"{method}_crossgen_results.json", "w") as f:
        json.dump({"per_gen": results, **macro,
                   "ckpt_val_auroc": float(ck["val_auroc"])}, f, indent=2)
    print(f"\n  MACRO  AUROC={macro['macro_auroc']:.4f}  "
          f"AP={macro['macro_ap']:.4f}  mAcc={macro['macro_mAcc']:.4f}")
    return macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["cnnspot", "npr", "univfd", "all"],
                    default="all")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    print("[1/3] Building train/val/test splits (matched to DHSDv2)...")
    train_items, mj_val_items, mj_test_items = build_train_items()
    print(f"  Train: {len(train_items):,}  MJ-val: {len(mj_val_items):,}  "
          f"MJ-test: {len(mj_test_items):,}")

    print("[2/3] Building cross-gen test split...")
    crossgen_val, crossgen_test = build_crossgen_split()
    n_test = sum(len(v) for v in crossgen_test.values())
    print(f"  Cross-gen generators: {len(crossgen_test)}")
    print(f"  Total cross-gen test images: {n_test:,}")

    methods = (["cnnspot", "npr", "univfd"]
               if args.method == "all" else [args.method])
    summary = {}

    print("[3/3] Training and evaluating...")
    for m in methods:
        ckpt = OUT_DIR / f"{m}_best.pt"
        if not args.eval_only:
            ckpt = train_one(m, train_items, mj_val_items)
        elif not ckpt.exists():
            print(f"  [warn] --eval-only set but {ckpt} not found, skipping")
            continue
        summary[m] = evaluate(m, ckpt, crossgen_test)

    print("\n\n" + "="*70)
    print("  FINAL COMPARISON")
    print("="*70)
    print(f"{'Method':<15}  {'AUROC':>7}  {'AP':>7}  {'mAcc':>7}")
    print("-" * 45)
    print(f"{'DHSDv2 (ours)':<15}  {0.8288:>7.4f}  {0.8394:>7.4f}  {0.6910:>7.4f}")
    for m, s in summary.items():
        print(f"{m.upper():<15}  {s['macro_auroc']:>7.4f}  "
              f"{s['macro_ap']:>7.4f}  {s['macro_mAcc']:>7.4f}")
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()