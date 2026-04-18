import os, json, time, sys, re, gc
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

BASE = Path("/NAS_DISK/Saptarshi_data")
HF_CACHE = str(BASE / "hf_cache")
OUTPUT_DIR = BASE / "gan_vlm_results"
OUTPUT_DIR.mkdir(exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

GAN_DIRS = {
    "starGAN": BASE / "dataset" / "fake" / "gan_test" / "starGAN",
    "BigGAN":  BASE / "dataset" / "fake" / "gan_test" / "BigGAN",
    "styleGAN": BASE / "dataset" / "fake" / "gan_test" / "styleGAN",
}
REAL_DIR = BASE / "dataset" / "real" / "coco"

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

def load_test_images(max_real=1000):
    images = []
    for gen_name, gen_dir in GAN_DIRS.items():
        for p in sorted(gen_dir.glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                images.append({"path": str(p), "id": p.stem, "ground_truth": "FAKE", "generator": gen_name})
    real_files = sorted([p for p in REAL_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
    import random; random.seed(42)
    for p in random.sample(real_files, min(max_real, len(real_files))):
        images.append({"path": str(p), "id": p.stem, "ground_truth": "REAL", "generator": "COCO"})
    return images

def parse_response(text):
    caption, cues = "", ["", "", ""]
    cap_match = re.search(r"CAPTION:\s*(.+)", text, re.IGNORECASE)
    if cap_match: caption = cap_match.group(1).strip()[:150]
    for num_str, content in re.findall(r"^\s*(\d)[.):]\s*(.+)", text, re.MULTILINE):
        idx = int(num_str) - 1
        if 0 <= idx < 3: cues[idx] = content.strip().replace("**", "")[:200]
    if not caption:
        for line in text.strip().split("\n"):
            line = line.strip()
            if line and not re.match(r"^\d[.)]", line): caption = line[:150]; break
    return caption, cues

def save_results(results, model_name):
    out = OUTPUT_DIR / f"gan_{model_name}.json"
    with open(out, "w") as f: json.dump({"results": results}, f)

def load_existing(model_name):
    out = OUTPUT_DIR / f"gan_{model_name}.json"
    if out.exists():
        with open(out) as f: return json.load(f).get("results", [])
    return []

# ── InternVL ──
def run_internvl(images):
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from PIL import Image
    MODEL = "OpenGVLab/InternVL2_5-8B-MPO"
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    def build_transform(sz=448):
        return T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((sz, sz), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    def dynamic_preprocess(image, max_num=6, image_size=448):
        w, h = image.size; aspect = w / h
        target_ratios = set()
        for n in range(1, max_num + 1):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if 1 <= i * j <= max_num: target_ratios.add((i, j))
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        best, best_diff = (1, 1), float("inf")
        for ratio in target_ratios:
            diff = abs(aspect - ratio[0] / ratio[1])
            if diff < best_diff: best_diff, best = diff, ratio
        tw, th = best[0] * image_size, best[1] * image_size
        resized = image.resize((tw, th), Image.LANCZOS); tiles = []
        for i in range(best[0] * best[1]):
            col, row = i % best[0], i // best[0]
            tiles.append(resized.crop((col*image_size, row*image_size, (col+1)*image_size, (row+1)*image_size)))
        if len(tiles) > 1: tiles.append(image.resize((image_size, image_size), Image.LANCZOS))
        return tiles
    def load_img(path):
        img = Image.open(path).convert("RGB"); transform = build_transform()
        return torch.stack([transform(t) for t in dynamic_preprocess(img)]).to(torch.bfloat16).cuda()
    print("  Loading InternVL2.5-8B-MPO...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, cache_dir=HF_CACHE)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", cache_dir=HF_CACHE).eval()
    gen_cfg = dict(max_new_tokens=256, do_sample=False)
    existing = load_existing("InternVL"); done = {(r["image_id"], r["assumption"]) for r in existing}; results = existing.copy()
    pbar = tqdm(images, desc="  InternVL", unit="img")
    for img_info in pbar:
        try: pv = load_img(img_info["path"])
        except: continue
        for assume in ("REAL", "FAKE"):
            if (img_info["id"], assume) in done: continue
            try:
                prompt = PROMPT_REAL if assume == "REAL" else PROMPT_FAKE
                with torch.inference_mode():
                    resp, _ = model.chat(tokenizer, pv, f"<image>\n{prompt}", generation_config=gen_cfg, history=None, return_history=True)
                caption, cues = parse_response(resp)
                results.append({"image_id": img_info["id"], "ground_truth": img_info["ground_truth"], "generator": img_info["generator"],
                    "model": "InternVL2.5-8B-MPO", "assumption": assume, "caption": caption, "cue_1": cues[0], "cue_2": cues[1], "cue_3": cues[2], "success": True})
                done.add((img_info["id"], assume))
            except: pass
        del pv; pbar.set_postfix(done=len(results))
        if len(results) % 500 == 0: save_results(results, "InternVL"); torch.cuda.empty_cache()
    save_results(results, "InternVL"); del model, tokenizer; gc.collect(); torch.cuda.empty_cache()

# ── Qwen ──
def run_qwen(images):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from PIL import Image
    MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
    print("  Loading Qwen2.5-VL-7B...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=HF_CACHE).eval()
    processor = AutoProcessor.from_pretrained(MODEL, cache_dir=HF_CACHE)
    existing = load_existing("Qwen"); done = {(r["image_id"], r["assumption"]) for r in existing}; results = existing.copy()
    pbar = tqdm(images, desc="  Qwen2.5-VL", unit="img")
    for img_info in pbar:
        try: img = Image.open(img_info["path"]).convert("RGB")
        except: continue
        for assume in ("REAL", "FAKE"):
            if (img_info["id"], assume) in done: continue
            try:
                prompt = PROMPT_REAL if assume == "REAL" else PROMPT_FAKE
                messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                resp = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                caption, cues = parse_response(resp)
                results.append({"image_id": img_info["id"], "ground_truth": img_info["ground_truth"], "generator": img_info["generator"],
                    "model": "Qwen2.5-VL-7B-Instruct", "assumption": assume, "caption": caption, "cue_1": cues[0], "cue_2": cues[1], "cue_3": cues[2], "success": True})
                done.add((img_info["id"], assume))
            except: pass
        pbar.set_postfix(done=len(results))
        if len(results) % 500 == 0: save_results(results, "Qwen"); torch.cuda.empty_cache()
    save_results(results, "Qwen"); del model, processor; gc.collect(); torch.cuda.empty_cache()

# ── Pixtral ──
def run_pixtral(images):
    import torch
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from PIL import Image
    MODEL = "mistral-community/pixtral-12b"
    print("  Loading Pixtral-12B...")
    model = LlavaForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=HF_CACHE).eval()
    processor = AutoProcessor.from_pretrained(MODEL, cache_dir=HF_CACHE)
    existing = load_existing("Pixtral"); done = {(r["image_id"], r["assumption"]) for r in existing}; results = existing.copy()
    pbar = tqdm(images, desc="  Pixtral", unit="img")
    for img_info in pbar:
        try: img = Image.open(img_info["path"]).convert("RGB")
        except: continue
        for assume in ("REAL", "FAKE"):
            if (img_info["id"], assume) in done: continue
            try:
                prompt = PROMPT_REAL if assume == "REAL" else PROMPT_FAKE
                conv = f"[INST]{prompt}\n[IMG][/INST]"
                inputs = processor(text=conv, images=[img], return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                resp = processor.decode(ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                caption, cues = parse_response(resp)
                results.append({"image_id": img_info["id"], "ground_truth": img_info["ground_truth"], "generator": img_info["generator"],
                    "model": "Pixtral-12B", "assumption": assume, "caption": caption, "cue_1": cues[0], "cue_2": cues[1], "cue_3": cues[2], "success": True})
                done.add((img_info["id"], assume))
            except: pass
        pbar.set_postfix(done=len(results))
        if len(results) % 500 == 0: save_results(results, "Pixtral"); torch.cuda.empty_cache()
    save_results(results, "Pixtral"); del model, processor; gc.collect(); torch.cuda.empty_cache()

# ── Phi-4 ──
def run_phi4(images):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from PIL import Image
    MODEL = "microsoft/Phi-4-multimodal-instruct"
    print("  Loading Phi-4-multimodal...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", cache_dir=HF_CACHE, attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, cache_dir=HF_CACHE)
    existing = load_existing("Phi4"); done = {(r["image_id"], r["assumption"]) for r in existing}; results = existing.copy()
    pbar = tqdm(images, desc="  Phi-4", unit="img")
    for img_info in pbar:
        try: img = Image.open(img_info["path"]).convert("RGB")
        except: continue
        for assume in ("REAL", "FAKE"):
            if (img_info["id"], assume) in done: continue
            try:
                prompt = PROMPT_REAL if assume == "REAL" else PROMPT_FAKE
                messages = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]
                text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=text, images=[img], return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    ids = model.generate(**inputs, max_new_tokens=256, do_sample=False, eos_token_id=processor.tokenizer.eos_token_id)
                resp = processor.tokenizer.decode(ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                caption, cues = parse_response(resp)
                results.append({"image_id": img_info["id"], "ground_truth": img_info["ground_truth"], "generator": img_info["generator"],
                    "model": "Phi-4-multimodal", "assumption": assume, "caption": caption, "cue_1": cues[0], "cue_2": cues[1], "cue_3": cues[2], "success": True})
                done.add((img_info["id"], assume))
            except: pass
        pbar.set_postfix(done=len(results))
        if len(results) % 500 == 0: save_results(results, "Phi4"); torch.cuda.empty_cache()
    save_results(results, "Phi4"); del model, processor; gc.collect(); torch.cuda.empty_cache()

# ── GLM-4V ──
def run_glm4v(images):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from PIL import Image
    MODEL = "THUDM/glm-4v-9b"
    print("  Loading GLM-4V-9B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, cache_dir=HF_CACHE)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", cache_dir=HF_CACHE).eval()
    existing = load_existing("GLM4V"); done = {(r["image_id"], r["assumption"]) for r in existing}; results = existing.copy()
    pbar = tqdm(images, desc="  GLM-4V", unit="img")
    for img_info in pbar:
        try: img = Image.open(img_info["path"]).convert("RGB")
        except: continue
        for assume in ("REAL", "FAKE"):
            if (img_info["id"], assume) in done: continue
            try:
                prompt = PROMPT_REAL if assume == "REAL" else PROMPT_FAKE
                inputs = tokenizer.apply_chat_template([{"role": "user", "image": img, "content": prompt}],
                    add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True).to(model.device)
                with torch.inference_mode():
                    ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                resp = tokenizer.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                caption, cues = parse_response(resp)
                results.append({"image_id": img_info["id"], "ground_truth": img_info["ground_truth"], "generator": img_info["generator"],
                    "model": "GLM-4V-9B", "assumption": assume, "caption": caption, "cue_1": cues[0], "cue_2": cues[1], "cue_3": cues[2], "success": True})
                done.add((img_info["id"], assume))
            except: pass
        pbar.set_postfix(done=len(results))
        if len(results) % 500 == 0: save_results(results, "GLM4V"); torch.cuda.empty_cache()
    save_results(results, "GLM4V"); del model, tokenizer; gc.collect(); torch.cuda.empty_cache()

# ── MAIN ──
if __name__ == "__main__":
    print("=" * 60)
    print("GAN Cross-Generator VLM Runner")
    print(f"Started: {datetime.now()}")
    print("=" * 60)
    images = load_test_images(max_real=1000)
    from collections import Counter
    print(f"Total: {len(images)} images — {dict(Counter(i['generator'] for i in images))}")
    MODELS = [("InternVL", run_internvl), ("Qwen", run_qwen), ("Pixtral", run_pixtral), ("Phi4", run_phi4), ("GLM4V", run_glm4v)]
    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        MODELS = [(n, f) for n, f in MODELS if target in n.lower()]
    for name, runner in MODELS:
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        t0 = time.time()
        try:
            runner(images); print(f"  {name} done in {(time.time()-t0)/60:.1f} min")
        except Exception as e:
            print(f"  {name} FAILED: {e}"); import traceback; traceback.print_exc()
    print(f"\nALL DONE — {datetime.now()}")
