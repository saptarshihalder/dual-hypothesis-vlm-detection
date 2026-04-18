#!/usr/bin/env python3
"""
Unified Dual Hypothesis VLM Runner
===================================
Runs all 6 VLMs with dual-hypothesis prompting on WildFake diffusion dataset.
Usage:
    python3 dual_vlm_runner.py --model internvl
    python3 dual_vlm_runner.py --model all        # sequential, all models
    python3 dual_vlm_runner.py --model pixtral --resume
"""

import os, json, time, torch, gc, sys, re, argparse
from pathlib import Path
from datetime import datetime
from PIL import Image

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
DATASET_BASE = "/NAS_DISK/Saptarshi_data/dataset"
HF_CACHE     = "/NAS_DISK/Saptarshi_data/hf_cache"
OUTPUT_DIR   = "/NAS_DISK/Saptarshi_data/results"
BACKUP_DIR   = "/NAS_DISK/Saptarshi_data/results_backup"

os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["TORCH_HOME"] = HF_CACHE
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ════════════════════════════════════════════════════════════
MODELS = {
    "internvl": {
        "name": "InternVL2.5-8B-MPO",
        "hf_path": "OpenGVLab/InternVL2_5-8B-MPO",
        "short": "internvl",
    },
    "pixtral": {
        "name": "Pixtral-12B",
        "hf_path": "mistral-community/pixtral-12b",
        "short": "pixtral",
    },
    "phi4": {
        "name": "Phi-4-multimodal",
        "hf_path": "microsoft/Phi-4-multimodal-instruct",
        "short": "phi4",
    },
    "glm4v": {
        "name": "GLM-4V-9B",
        "hf_path": "THUDM/glm-4v-9b",
        "short": "glm4v",
    },
    "gemma3": {
        "name": "Gemma-3-12B-IT",
        "hf_path": "google/gemma-3-12b-it",
        "short": "gemma3",
    },
    "qwen3vl": {
        "name": "Qwen2.5-VL-7B-Instruct",
        "hf_path": "Qwen/Qwen2.5-VL-7B-Instruct",
        "short": "qwen3vl",
    },
}

# ════════════════════════════════════════════════════════════
# PROMPTS
# ════════════════════════════════════════════════════════════
PROMPT_REAL = """This is a real photograph. Write exactly 4 lines, no blanks allowed:
CAPTION: <one sentence describing the scene>
1. <in at least 10 words, describe one visual reason why this image looks real>
2. <in at least 10 words, describe another visual reason why this image looks real>
3. <in at least 10 words, describe another visual reason why this image looks real>"""

PROMPT_FAKE = """This is AI-generated. Write exactly 4 lines, no blanks allowed:
CAPTION: <one sentence describing the scene>
1. <in at least 10 words, describe one visual reason why this image looks fake or artificial>
2. <in at least 10 words, describe another visual reason why this image looks fake or artificial>
3. <in at least 10 words, describe another visual reason why this image looks fake or artificial>"""

# ════════════════════════════════════════════════════════════
# TERMINAL COLORS
# ════════════════════════════════════════════════════════════
C = {
    "r": "\033[91m", "g": "\033[92m", "y": "\033[93m",
    "b": "\033[94m", "c": "\033[96m", "m": "\033[95m",
    "x": "\033[0m",  "B": "\033[1m",  "d": "\033[2m",
    "bg_g": "\033[42m", "bg_r": "\033[41m", "bg_b": "\033[44m",
    "w": "\033[97m",
}

# ════════════════════════════════════════════════════════════
# ASCII IMAGE PREVIEW (colored, fast)
# ════════════════════════════════════════════════════════════
def print_ascii_image(image_path, width=60):
    try:
        img = Image.open(image_path).convert("RGB")
        aspect = img.height / img.width
        height = int(width * aspect * 0.42)
        height = max(height, 5)
        img = img.resize((width, height), Image.NEAREST)
        chars = " .:-=+*#%@"
        pixels = list(img.getdata())
        buf = [f"  {C['d']}┌{'─' * width}┐{C['x']}"]
        for y in range(height):
            row = [f"  {C['d']}│{C['x']}"]
            off = y * width
            for x in range(width):
                r, g, b = pixels[off + x]
                brightness = (r + g + b) // 3
                ci = min(brightness * len(chars) // 256, len(chars) - 1)
                # Pick closest ANSI color
                if r > g + 30 and r > b + 30:         cc = "\033[91m"
                elif g > r + 30 and g > b + 30:       cc = "\033[92m"
                elif b > r + 30 and b > g + 30:       cc = "\033[94m"
                elif r > 180 and g > 180 and b < 120: cc = "\033[93m"
                elif r > 180 and b > 180 and g < 120: cc = "\033[95m"
                elif g > 180 and b > 180 and r < 120: cc = "\033[96m"
                elif brightness > 200:                  cc = "\033[97m"
                elif brightness < 50:                   cc = "\033[90m"
                else:                                   cc = "\033[37m"
                row.append(f"{cc}{chars[ci]}")
            row.append(f"{C['x']}{C['d']}│{C['x']}")
            buf.append("".join(row))
        buf.append(f"  {C['d']}└{'─' * width}┘{C['x']}")
        print("\n".join(buf))
    except Exception as e:
        print(f"  {C['r']}[preview err: {e}]{C['x']}")


# ════════════════════════════════════════════════════════════
# RESPONSE PARSING
# ════════════════════════════════════════════════════════════
def parse_response(text):
    caption = ""
    cues = ["", "", ""]
    cap_match = re.search(r"CAPTION:\s*(.+)", text, re.IGNORECASE)
    if cap_match:
        caption = cap_match.group(1).strip()[:200]
    cue_matches = re.findall(r"^\s*(\d)[.):]\s*(.+)", text, re.MULTILINE)
    for num_str, content in cue_matches:
        idx = int(num_str) - 1
        if 0 <= idx < 3:
            cues[idx] = content.strip().replace("**", "")[:250]
    if not caption:
        for line in text.strip().split("\n"):
            line = line.strip()
            if line and not re.match(r"^\d[.)]", line):
                caption = line[:200]
                break
    if not any(cues):
        words = text.split()
        cs = max(len(words) // 4, 1)
        for i in range(3):
            cues[i] = " ".join(words[cs * (i + 1):cs * (i + 2)])[:250]
    return caption, cues


# ════════════════════════════════════════════════════════════
# DATASET LOADING
# ════════════════════════════════════════════════════════════
def load_images():
    imgs = []
    fake_dir = Path(DATASET_BASE) / "fake" / "midjourney"
    real_dir = Path(DATASET_BASE) / "real" / "coco"

    for label, d in [("REAL", real_dir), ("FAKE", fake_dir)]:
        if not d.exists():
            print(f"  {C['r']}WARNING: {d} not found!{C['x']}")
            continue
        for p in sorted(d.glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                imgs.append({"path": str(p), "id": p.stem, "ground_truth": label})

    real_count = sum(1 for i in imgs if i["ground_truth"] == "REAL")
    fake_count = sum(1 for i in imgs if i["ground_truth"] == "FAKE")
    print(f"  {C['g']}REAL: {real_count}{C['x']}  |  {C['r']}FAKE: {fake_count}{C['x']}  |  Total: {len(imgs)}")
    return imgs


# ════════════════════════════════════════════════════════════
# JSON I/O WITH CHECKPOINTING
# ════════════════════════════════════════════════════════════
def load_results(fp):
    try:
        with open(fp) as f:
            d = json.load(f)
        print(f"  Loaded {len(d.get('results', []))} existing entries from {fp}")
        return d
    except Exception:
        return {"results": [], "metadata": {"created": datetime.now().isoformat()}}


def save_results(data, fp):
    try:
        tmp = fp + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, fp)  # atomic write
    except Exception as e:
        print(f"  {C['r']}Save error: {e}{C['x']}")


# ════════════════════════════════════════════════════════════
# DISPLAY RESULTS
# ════════════════════════════════════════════════════════════
def print_dual_result(model_name, img_info, real_cap, real_cues, fake_cap, fake_cues, count, total, rate, show_ascii=True):
    gt = img_info["ground_truth"]
    gtc = C["g"] if gt == "REAL" else C["r"]
    gt_bg = C["bg_g"] if gt == "REAL" else C["bg_r"]

    print(f"\n{'═' * 70}")
    print(f"  {C['B']}{C['c']}[{model_name}]{C['x']}  "
          f"{C['B']}{count}/{total}{C['x']}  "
          f"{C['c']}{rate:.1f} img/min{C['x']}  "
          f"ETA: {C['y']}{(total - count) / max(rate, 0.1):.0f}m{C['x']}")
    print(f"  Image: {C['y']}{img_info['id']}{C['x']}  "
          f"Ground Truth: {gt_bg}{C['w']} {gt} {C['x']}")
    print()

    # ASCII preview (skip for speed unless show_ascii)
    if show_ascii:
        print_ascii_image(img_info["path"], width=60)
        print()

    # Assume REAL results
    print(f"  {C['bg_g']}{C['w']} ASSUME REAL {C['x']}")
    print(f"  {C['B']}Caption:{C['x']} {real_cap[:80]}")
    for i, cue in enumerate(real_cues, 1):
        print(f"  {C['g']}Cue {i}:{C['x']} {cue[:70]}")
    print()

    # Assume FAKE results
    print(f"  {C['bg_r']}{C['w']} ASSUME FAKE {C['x']}")
    print(f"  {C['B']}Caption:{C['x']} {fake_cap[:80]}")
    for i, cue in enumerate(fake_cues, 1):
        print(f"  {C['r']}Cue {i}:{C['x']} {cue[:70]}")

    sys.stdout.flush()


# ════════════════════════════════════════════════════════════
# INTERNVL2.5-8B-MPO
# ════════════════════════════════════════════════════════════
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _internvl_build_transform(input_size=448):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def _internvl_dynamic_preprocess(image, min_num=1, max_num=6, image_size=448):
    w, h = image.size
    aspect = w / h
    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if min_num <= i * j <= max_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best, best_diff = (1, 1), float("inf")
    area = w * h
    for ratio in target_ratios:
        diff = abs(aspect - ratio[0] / ratio[1])
        if diff < best_diff or (diff == best_diff and
            abs(area - ratio[0]*ratio[1]*image_size*image_size) <
            abs(area - best[0]*best[1]*image_size*image_size)):
            best_diff, best = diff, ratio

    tw, th = best[0] * image_size, best[1] * image_size
    resized = image.resize((tw, th), Image.LANCZOS)
    tiles = []
    for i in range(best[0] * best[1]):
        col, row = i % best[0], i // best[0]
        tiles.append(resized.crop((col*image_size, row*image_size,
                                   (col+1)*image_size, (row+1)*image_size)))
    if len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size), Image.LANCZOS))
    return tiles

def _internvl_load_image(path):
    img = Image.open(path).convert("RGB")
    transform = _internvl_build_transform(448)
    tiles = _internvl_dynamic_preprocess(img, max_num=6)
    pv = torch.stack([transform(t) for t in tiles])
    return pv.to(torch.bfloat16).cuda()

def load_internvl():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = MODELS["internvl"]["hf_path"]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, cache_dir=HF_CACHE)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto", cache_dir=HF_CACHE, attn_implementation="eager")
    model.eval()
    return model, tok

def infer_internvl(model, tok, image_path, prompt):
    pv = _internvl_load_image(image_path)
    gen_cfg = dict(max_new_tokens=128, do_sample=False)
    with torch.inference_mode():
        resp, _ = model.chat(tok, pv, f"<image>\n{prompt}",
                             generation_config=gen_cfg, history=None, return_history=True)
    del pv
    return resp


# ════════════════════════════════════════════════════════════
# PIXTRAL-12B
# ════════════════════════════════════════════════════════════
def load_pixtral():
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    path = MODELS["pixtral"]["hf_path"]
    model = LlavaForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=HF_CACHE, attn_implementation="eager")
    processor = AutoProcessor.from_pretrained(path, cache_dir=HF_CACHE)
    model.eval()
    return model, processor

def infer_pixtral(model, processor, image_path, prompt):
    img = Image.open(image_path).convert("RGB")
    conversation = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=text, images=[img], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    resp = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    del inputs
    return resp


# ════════════════════════════════════════════════════════════
# PHI-4-MULTIMODAL
# ════════════════════════════════════════════════════════════
def load_phi4():
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
    path = MODELS["phi4"]["hf_path"]
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, cache_dir=HF_CACHE)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True, cache_dir=HF_CACHE)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto", cache_dir=HF_CACHE,
        )
    model.eval()
    return model, processor

def infer_phi4(model, processor, image_path, prompt):
    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]
    text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[img], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                             eos_token_id=processor.tokenizer.eos_token_id)
    resp = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    del inputs
    return resp


# ════════════════════════════════════════════════════════════
# GLM-4V-9B
# ════════════════════════════════════════════════════════════
def load_glm4v():
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    path = MODELS["glm4v"]["hf_path"]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, cache_dir=HF_CACHE)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True, cache_dir=HF_CACHE)
    config.max_length = getattr(config, "max_length", 8192)
    config.seq_length = getattr(config, "seq_length", 8192)
    model = AutoModelForCausalLM.from_pretrained(
        path, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, cache_dir=HF_CACHE
    ).to("cuda").eval()
    return model, tok

def infer_glm4v(model, tok, image_path, prompt):
    img = Image.open(image_path).convert("RGB")
    inputs = tok.apply_chat_template(
        [{"role": "user", "image": img, "content": prompt}],
        add_generation_prompt=True, tokenize=True, return_tensors="pt",
        return_dict=True).to("cuda")
    gen_kwargs = {"max_length": 2500, "do_sample": False, "top_k": 1}
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    out = out[:, inputs["input_ids"].shape[1]:]
    resp = tok.decode(out[0], skip_special_tokens=True)
    del inputs
    return resp


# ════════════════════════════════════════════════════════════
# GEMMA-3-27B-IT
# ════════════════════════════════════════════════════════════
def load_gemma3():
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    path = MODELS["gemma3"]["hf_path"]
    processor = AutoProcessor.from_pretrained(path, cache_dir=HF_CACHE)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=HF_CACHE)
    model.eval()
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile enabled!")
    except:
        pass
    return model, processor

def infer_gemma3(model, processor, image_path, prompt):
    img = Image.open(image_path).convert("RGB")
    # Resize large images to speed up processing
    max_size = 512
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        img = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)), Image.BILINEAR)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=[img], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    resp = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    del inputs
    return resp


# ════════════════════════════════════════════════════════════
# QWEN2.5-VL-7B
# ════════════════════════════════════════════════════════════
def load_qwen3vl():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    path = MODELS["qwen3vl"]["hf_path"]
    processor = AutoProcessor.from_pretrained(path, cache_dir=HF_CACHE)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto",
        cache_dir=HF_CACHE)
    model.eval()
    return model, processor

def infer_qwen3vl(model, processor, image_path, prompt):
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "image", "image": f"file://{image_path}"},
        {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    resp = processor.batch_decode(out[:, inputs["input_ids"].shape[-1]:],
                                  skip_special_tokens=True)[0]
    del inputs
    return resp


# ════════════════════════════════════════════════════════════
# DISPATCHER
# ════════════════════════════════════════════════════════════
LOADERS = {
    "internvl": load_internvl,
    "pixtral":  load_pixtral,
    "phi4":     load_phi4,
    "glm4v":    load_glm4v,
    "gemma3":   load_gemma3,
    "qwen3vl":  load_qwen3vl,
}

# For models that use (model, tokenizer) vs (model, processor),
# the infer functions all take (model, tok_or_proc, image_path, prompt)
INFERENCERS = {
    "internvl": infer_internvl,
    "pixtral":  infer_pixtral,
    "phi4":     infer_phi4,
    "glm4v":    infer_glm4v,
    "gemma3":   infer_gemma3,
    "qwen3vl":  infer_qwen3vl,
}


# ════════════════════════════════════════════════════════════
# MAIN PROCESSING LOOP
# ════════════════════════════════════════════════════════════
def run_model(model_key):
    info = MODELS[model_key]
    model_name = info["name"]
    output_file = os.path.join(OUTPUT_DIR, f"{model_key}_results.json")
    backup_file = os.path.join(BACKUP_DIR, f"{model_key}_backup.json")

    print(f"\n{'═' * 70}")
    print(f"  {C['B']}{C['c']}  {model_name}  {C['x']}")
    print(f"  {C['d']}HF: {info['hf_path']}{C['x']}")
    print(f"  {C['d']}Output: {output_file}{C['x']}")
    print(f"  {C['d']}Started: {datetime.now()}{C['x']}")
    print(f"{'═' * 70}")

    # Load existing results
    data = load_results(output_file)
    processed = {(r["image_id"], r["assumption"]) for r in data["results"]}

    # Load dataset
    images = load_images()
    total_target = len(images) * 2  # each image × 2 assumptions

    already = len(processed)
    remaining = total_target - already
    print(f"  Already done: {already}/{total_target}")
    print(f"  Remaining:    {remaining}")

    if remaining <= 0:
        print(f"  {C['g']}Already complete for {model_name}!{C['x']}")
        return

    # Load model
    print(f"\n  Loading {model_name}...")
    t_load = time.time()
    try:
        model, tok_or_proc = LOADERS[model_key]()
    except Exception as e:
        print(f"  {C['r']}FAILED to load {model_name}: {e}{C['x']}")
        return
    print(f"  {C['g']}Loaded in {time.time()-t_load:.0f}s | "
          f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB{C['x']}")

    infer_fn = INFERENCERS[model_key]
    t0 = time.time()
    count = 0
    errors = 0
    save_interval = 200

    for img_idx, img_info in enumerate(images):
        img_id = img_info["id"]

        # Skip if both assumptions already done
        if (img_id, "REAL") in processed and (img_id, "FAKE") in processed:
            continue

        # Run both assumptions
        real_cap, real_cues = "", ["", "", ""]
        fake_cap, fake_cues = "", ["", "", ""]

        for assume in ("REAL", "FAKE"):
            key = (img_id, assume)
            if key in processed:
                continue

            prompt = PROMPT_REAL if assume == "REAL" else PROMPT_FAKE

            try:
                resp = infer_fn(model, tok_or_proc, img_info["path"], prompt)
                caption, cues = parse_response(resp)

                data["results"].append({
                    "image_id": img_id,
                    "ground_truth": img_info["ground_truth"],
                    "model": model_name,
                    "assumption": assume,
                    "caption": caption,
                    "cue_1": cues[0],
                    "cue_2": cues[1],
                    "cue_3": cues[2],
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                })
                processed.add(key)
                count += 1

                if assume == "REAL":
                    real_cap, real_cues = caption, cues
                else:
                    fake_cap, fake_cues = caption, cues

            except Exception as e:
                errors += 1
                print(f"  {C['r']}Error {img_id}/{assume}: {e}{C['x']}")
                if errors > 300:
                    print(f"  {C['r']}Too many errors ({errors}), stopping.{C['x']}")
                    break

        # Display after both assumptions done
        if count > 0:
            elapsed = time.time() - t0
            rate = count / (elapsed / 60) if elapsed > 0 else 0
            show_full = (count % 20 == 0)  # ASCII every 10th image
            print_dual_result(model_name, img_info,
                              real_cap, real_cues, fake_cap, fake_cues,
                              len(processed), total_target, rate, show_ascii=show_full)

        # Checkpoint
        if count > 0 and count % save_interval == 0:
            save_results(data, output_file)
            save_results(data, backup_file)
            elapsed = time.time() - t0
            rate = count / (elapsed / 60) if elapsed > 0 else 0
            eta_min = (total_target - len(processed)) / rate if rate > 0 else 0
            print(f"\n  {C['c']}{'─' * 50}")
            print(f"  SAVED! {len(data['results'])} entries | "
                  f"Errors: {errors} | "
                  f"ETA: {eta_min:.0f}m ({eta_min/60:.1f}h)")
            print(f"  {'─' * 50}{C['x']}")

        # Memory cleanup
        if count % 25 == 0:
            torch.cuda.empty_cache()

        if errors > 300:
            break

    # Final save
    save_results(data, output_file)
    save_results(data, backup_file)
    hrs = (time.time() - t0) / 3600
    print(f"\n  {C['g']}{'═' * 50}")
    print(f"  {model_name} DONE!")
    print(f"  Total entries: {len(data['results'])}")
    print(f"  Time: {hrs:.1f}h | Errors: {errors}")
    print(f"  {'═' * 50}{C['x']}")

    # Free GPU
    del model, tok_or_proc
    gc.collect()
    torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Dual Hypothesis VLM Runner")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODELS.keys()) + ["all"],
                        help="Which VLM to run (or 'all' for sequential)")
    args = parser.parse_args()

    print(f"{C['B']}{C['c']}")
    print(f"  ╔══════════════════════════════════════════════════╗")
    print(f"  ║   Dual Hypothesis VLM Runner v2.0               ║")
    print(f"  ║   WildFake Diffusion — Midjourney + COCO        ║")
    print(f"  ╚══════════════════════════════════════════════════╝{C['x']}")
    print(f"  {C['d']}Time: {datetime.now()}{C['x']}")
    print(f"  {C['d']}Dataset: {DATASET_BASE}{C['x']}")
    print(f"  {C['d']}GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}{C['x']}")
    print(f"  {C['d']}VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB{C['x']}" if torch.cuda.is_available() else "")

    if args.model == "all":
        order = ["internvl", "phi4", "glm4v", "qwen3vl", "gemma3", "pixtral"]
        print(f"\n  Running ALL models: {', '.join(order)}")
        for m in order:
            run_model(m)
    else:
        run_model(args.model)

    print(f"\n  {C['g']}{C['B']}All done!{C['x']}")


if __name__ == "__main__":
    main()
