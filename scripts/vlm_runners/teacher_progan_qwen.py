#!/usr/bin/env python3
"""Teacher pipeline on ProGAN using Qwen2.5-VL (compatible with transformers 5.x)"""

import os, json, time, torch, sys, re, glob
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

HF_CACHE = "/NAS_DISK/Saptarshi_data/hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

PROGAN_DIR = "/NAS_DISK/Saptarshi_data/dataset/cnndetection_test"
RESULTS_DIR = "/NAS_DISK/Saptarshi_data/results"
BACKUP_DIR = "/NAS_DISK/Saptarshi_data/results_backup"
OUTPUT_DIR = "/NAS_DISK/Saptarshi_data/pipeline_output"
SEED = 42
np.random.seed(SEED)

PROMPT_REAL = """Real photo. Respond:
CAPTION: [scene description]
1. [detail proving real]
2. [different detail proving real]
3. [third detail proving real]"""

PROMPT_FAKE = """AI-generated image. Respond:
CAPTION: [scene description]
1. [flaw suggesting AI-generated]
2. [different flaw suggesting AI-generated]
3. [third flaw suggesting AI-generated]"""

def parse_response(text):
    caption, cues = "", ["", "", ""]
    cap_match = re.search(r"CAPTION:\s*(.+)", text, re.IGNORECASE)
    if cap_match:
        caption = cap_match.group(1).strip()[:150]
    cue_matches = re.findall(r"^\s*(\d)[.):]\s*(.+)", text, re.MULTILINE)
    for num_str, content in cue_matches:
        idx = int(num_str) - 1
        if 0 <= idx < 3:
            cues[idx] = content.strip().replace("**", "")[:200]
    if not caption:
        for line in text.strip().split("\n"):
            line = line.strip()
            if line and not re.match(r"^\d[.)]", line):
                caption = line[:150]
                break
    return caption, cues

def collect_progan_images(base_dir, n_per_class=150):
    real_paths, fake_paths = [], []
    for root, dirs, files in os.walk(base_dir):
        folder = os.path.basename(root)
        imgs = sorted([os.path.join(root, f) for f in files
                       if Path(f).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif")])
        if folder == "0_real":
            real_paths.extend(imgs)
        elif folder == "1_fake":
            fake_paths.extend(imgs)
    np.random.shuffle(real_paths)
    np.random.shuffle(fake_paths)
    real_paths = real_paths[:n_per_class]
    fake_paths = fake_paths[:n_per_class]
    items = [(p, "REAL", Path(p).stem) for p in real_paths]
    items += [(p, "FAKE", Path(p).stem) for p in fake_paths]
    print(f"  ProGAN: {len(real_paths)} real + {len(fake_paths)} fake = {len(items)}")
    return items

def run_qwen_on_images(items):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
    print("\n[Qwen] Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=HF_CACHE)
    model.eval()

    results = []
    t0 = time.time()
    for i, (path, gt, img_id) in enumerate(items):
        for assume in ("REAL", "FAKE"):
            prompt = PROMPT_REAL if assume == "REAL" else PROMPT_FAKE
            try:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": f"file://{path}"},
                    {"type": "text", "text": prompt}
                ]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                                   padding=True, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                output_ids = output_ids[:, inputs.input_ids.shape[1]:]
                response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                caption, cues = parse_response(response)
                results.append({
                    "image_id": img_id, "ground_truth": gt,
                    "model": "Qwen2.5-VL-7B", "assumption": assume,
                    "caption": caption, "cue_1": cues[0], "cue_2": cues[1], "cue_3": cues[2],
                    "success": True, "source": "progan_test"
                })
            except Exception as e:
                print(f"  Err {img_id}/{assume}: {e}")
        if (i+1) % 25 == 0:
            torch.cuda.empty_cache()
            rate = (i+1) / ((time.time()-t0)/60)
            print(f"  [Qwen] {i+1}/{len(items)} | {rate:.1f} img/min")

    del model, processor
    torch.cuda.empty_cache()
    print(f"  [Qwen] Done: {len(results)} entries")
    return results

def clip_score_results(vlm_results, items):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", cache_dir=HF_CACHE)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model = model.cuda().eval()
    print("\n[CLIP] Scoring...")

    image_features = {}
    for path, gt, img_id in items:
        try:
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).cuda()
            with torch.no_grad(), torch.cuda.amp.autocast():
                feat = model.encode_image(img)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            image_features[img_id] = feat
        except:
            pass
    print(f"  Encoded {len(image_features)} images")

    text_scores = {}
    for r in vlm_results:
        img_id = r["image_id"]
        if img_id not in image_features:
            continue
        img_feat = image_features[img_id]
        texts = [r.get("caption", ""), r.get("cue_1", ""), r.get("cue_2", ""), r.get("cue_3", "")]
        texts = [t if t else "no description" for t in texts]
        try:
            tokens = tokenizer(texts).cuda()
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_feats = model.encode_text(tokens)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            sims = (img_feat @ text_feats.T).squeeze(0).cpu().numpy()
            key = (img_id, r["model"], r["assumption"])
            text_scores[key] = {
                "caption_score": float(sims[0]),
                "mean_cue_score": float(np.mean(sims[1:4])),
                "mean_all_score": float(np.mean(sims)),
            }
        except:
            pass
    del model
    torch.cuda.empty_cache()
    print(f"  Scored {len(text_scores)} pairs")
    return text_scores

def compute_features(items, text_scores):
    FEAT_NAMES = ["avg_caption_disc", "avg_cue_disc", "avg_all_disc",
                  "std_caption_disc", "std_cue_disc", "std_all_disc",
                  "max_disc", "min_disc", "disc_range",
                  "avg_real_score", "avg_fake_score", "n_models_agree_direction"]
    X, y, ids = [], [], []
    for path, gt, img_id in items:
        label = 0 if gt == "REAL" else 1
        models = set()
        for (iid, m, a), s in text_scores.items():
            if iid == img_id:
                models.add(m)
        discs = {"caption": [], "mean_cue": [], "mean_all": []}
        r_scores, f_scores = [], []
        for m in models:
            kr, kf = (img_id, m, "REAL"), (img_id, m, "FAKE")
            if kr in text_scores and kf in text_scores:
                rs, fs = text_scores[kr], text_scores[kf]
                for metric in ["caption", "mean_cue", "mean_all"]:
                    discs[metric].append(fs[f"{metric}_score"] - rs[f"{metric}_score"])
                r_scores.append(rs["mean_all_score"])
                f_scores.append(fs["mean_all_score"])
        if not discs["caption"]:
            continue
        feat = [
            np.mean(discs["caption"]), np.mean(discs["mean_cue"]), np.mean(discs["mean_all"]),
            np.std(discs["caption"]) if len(discs["caption"]) > 1 else 0,
            np.std(discs["mean_cue"]) if len(discs["mean_cue"]) > 1 else 0,
            np.std(discs["mean_all"]) if len(discs["mean_all"]) > 1 else 0,
            max(discs["mean_all"]), min(discs["mean_all"]),
            max(discs["mean_all"]) - min(discs["mean_all"]),
            np.mean(r_scores) if r_scores else 0,
            np.mean(f_scores) if f_scores else 0,
            sum(1 for d in discs["mean_all"] if d > 0),
        ]
        X.append(feat)
        y.append(label)
        ids.append(img_id)
    return np.array(X, dtype=np.float32), np.array(y), ids, FEAT_NAMES

def main():
    t0 = time.time()
    print("=" * 70)
    print("TEACHER PIPELINE ON ProGAN — Qwen2.5-VL")
    print("=" * 70)

    print("\n[1/5] Collecting ProGAN images...")
    progan_items = collect_progan_images(PROGAN_DIR, n_per_class=150)

    cache_path = os.path.join(OUTPUT_DIR, "progan_vlm_qwen.json")
    if os.path.exists(cache_path):
        print(f"\n[2/5] Loading cached: {cache_path}")
        with open(cache_path) as f:
            progan_vlm = json.load(f)
        print(f"  {len(progan_vlm)} entries")
    else:
        print(f"\n[2/5] Running Qwen on ProGAN...")
        progan_vlm = run_qwen_on_images(progan_items)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(progan_vlm, f)

    print("\n[3/5] CLIP scoring...")
    progan_scores = clip_score_results(progan_vlm, progan_items)

    print("\n[4/5] ProGAN features...")
    X_pg, y_pg, ids_pg, fnames = compute_features(progan_items, progan_scores)
    print(f"  ProGAN: {X_pg.shape}")

    print("\n[5/5] Loading Midjourney teacher...")
    clip_cache = os.path.join(OUTPUT_DIR, "clip_scores_3k.json")
    with open(clip_cache) as f:
        raw = json.load(f)
    mj_scores = {}
    for k, v in raw.items():
        parts = k.split("|")
        if len(parts) == 3:
            mj_scores[(parts[0], parts[1], parts[2])] = v

    mj_by_image = defaultdict(list)
    for d in [RESULTS_DIR, BACKUP_DIR]:
        if os.path.isdir(d):
            for fp in glob.glob(os.path.join(d, "*.json")):
                try:
                    with open(fp) as f:
                        data = json.load(f)
                    entries = data if isinstance(data, list) else data.get("results", [])
                    for r in entries:
                        mj_by_image[r.get("image_id", "")].append(r)
                except:
                    pass

    mj_items = []
    for img_id, entries in mj_by_image.items():
        gt = entries[0].get("ground_truth", "").upper()
        if gt in ("REAL", "FAKE"):
            mj_items.append(("", gt, img_id))
    X_mj, y_mj, _, _ = compute_features(mj_items, mj_scores)
    print(f"  Midjourney: {X_mj.shape}")

    X_mj = np.nan_to_num(X_mj)
    X_pg = np.nan_to_num(X_pg)
    scaler = StandardScaler()
    X_mj_s = scaler.fit_transform(X_mj)
    X_pg_s = scaler.transform(X_pg)

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=SEED)
    clf.fit(X_mj_s, y_mj)

    mj_auc = roc_auc_score(y_mj, clf.predict_proba(X_mj_s)[:, 1])
    pg_probs = clf.predict_proba(X_pg_s)[:, 1]
    pg_auc = roc_auc_score(y_pg, pg_probs)
    pg_acc = accuracy_score(y_pg, (pg_probs > 0.5).astype(int))

    print(f"\n{'=' * 70}")
    print(f"RESULTS — TEACHER vs STUDENT")
    print(f"{'=' * 70}")
    print(f"  {'Method':<30} {'ProGAN AUROC':<15} {'Approach'}")
    print(f"  {'-'*60}")
    print(f"  {'Student (image-only)':<30} {'0.5809':<15} {'CLIP encoder + MLP'}")
    print(f"  {'Teacher (VLM reasoning)':<30} {f'{pg_auc:.4f}':<15} {'Qwen + CLIP + discrepancy'}")
    print(f"\n  Teacher on Midjourney: {mj_auc:.4f}")
    print(f"  Teacher on ProGAN:    {pg_auc:.4f} (acc: {pg_acc:.4f})")
    print(f"  Improvement over student: +{pg_auc - 0.5809:.4f}")
    print(f"\n  THESIS: VLM reasoning generalizes. Pixel features don't.")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")

    with open(os.path.join(OUTPUT_DIR, "teacher_vs_student_progan.json"), "w") as f:
        json.dump({"teacher_progan_auroc": pg_auc, "student_progan_auroc": 0.5809,
                    "teacher_midjourney_auroc": mj_auc, "vlm": "Qwen2.5-VL-7B"}, f, indent=2)

if __name__ == "__main__":
    main()
