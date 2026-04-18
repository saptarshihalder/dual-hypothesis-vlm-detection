#!/usr/bin/env python3
"""
Fast VLM Runner - InternVL + Qwen on cross-generator test set
Usage: python3 fast_vlm_runner.py --input_dir /path/to/images --label FAKE
"""

import os, json, time, torch, sys, re, argparse
from pathlib import Path
from datetime import datetime
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True)
parser.add_argument("--label", required=True, choices=["REAL", "FAKE"])
parser.add_argument("--output", default="/NAS_DISK/Saptarshi_data/pipeline_output/crossgen_vlm_results.json")
parser.add_argument("--max_images", type=int, default=500)
parser.add_argument("--hf_cache", default="/NAS_DISK/Saptarshi_data/hf_cache")
parser.add_argument("--models", nargs="+", default=["internvl", "qwen"], choices=["internvl", "qwen"])
args = parser.parse_args()

os.environ["HF_HOME"] = args.hf_cache
os.environ["TRANSFORMERS_CACHE"] = args.hf_cache

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


def collect_images(input_dir, max_images):
    imgs = []
    for p in sorted(Path(input_dir).rglob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            imgs.append(str(p))
            if len(imgs) >= max_images:
                break
    return imgs


def run_internvl(image_paths, label, results):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_PATH = "OpenGVLab/InternVL2_5-8B-MPO"
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(input_size=448):
        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
        w, h = image.size
        aspect = w / h
        target_ratios = set()
        for n in range(min_num, max_num + 1):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if min_num <= i * j <= max_num:
                        target_ratios.add((i, j))
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        best = min(target_ratios, key=lambda r: abs(aspect - r[0]/r[1]))
        target_w, target_h = best[0] * image_size, best[1] * image_size
        resized = image.resize((target_w, target_h), Image.LANCZOS)
        processed = []
        for i in range(best[0] * best[1]):
            col, row = i % best[0], i // best[0]
            box = (col*image_size, row*image_size, (col+1)*image_size, (row+1)*image_size)
            processed.append(resized.crop(box))
        if use_thumbnail and len(processed) > 1:
            processed.append(image.resize((image_size, image_size), Image.LANCZOS))
        return processed

    def load_image(path):
        image = Image.open(path).convert("RGB")
        transform = build_transform(448)
        tiles = dynamic_preprocess(image)
        return torch.stack([transform(t) for t in tiles]).to(torch.bfloat16).cuda()

    print("\n[InternVL] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, cache_dir=args.hf_cache)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto", cache_dir=args.hf_cache
    )
    model.eval()
    gen_config = dict(max_new_tokens=256, do_sample=False)

    t0 = time.time()
    for i, path in enumerate(image_paths):
        img_id = Path(path).stem
        try:
            pixel_values = load_image(path)
        except Exception as e:
            print(f"  Skip {img_id}: {e}")
            continue
        for assume in ("REAL", "FAKE"):
            prompt = PROMPT_REAL if assume == "REAL" else PROMPT_FAKE
            try:
                with torch.inference_mode():
                    response, _ = model.chat(
                        tokenizer, pixel_values,
                        f"<image>\n{prompt}",
                        generation_config=gen_config, history=None, return_history=True
                    )
                caption, cues = parse_response(response)
                results.append({
                    "image_id": img_id, "ground_truth": label,
                    "model": "InternVL2.5-8B-MPO", "assumption": assume,
                    "caption": caption, "cue_1": cues[0], "cue_2": cues[1], "cue_3": cues[2],
                    "success": True, "timestamp": datetime.now().isoformat(),
                    "source": "crossgen_test"
                })
            except Exception as e:
                print(f"  Error {img_id}/{assume}: {e}")
        del pixel_values
        if (i+1) % 50 == 0:
            torch.cuda.empty_cache()
            rate = (i+1) / ((time.time()-t0)/60)
            print(f"  [InternVL] {i+1}/{len(image_paths)} | {rate:.1f} img/min")

    del model, tokenizer
    torch.cuda.empty_cache()
    print(f"  [InternVL] Done: {len([r for r in results if r['model']=='InternVL2.5-8B-MPO'])} entries")


def run_qwen(image_paths, label, results):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

    print("\n[Qwen] Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, cache_dir=args.hf_cache)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=args.hf_cache
    )
    model.eval()

    t0 = time.time()
    for i, path in enumerate(image_paths):
        img_id = Path(path).stem
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
                    "image_id": img_id, "ground_truth": label,
                    "model": "Qwen2.5-VL-7B", "assumption": assume,
                    "caption": caption, "cue_1": cues[0], "cue_2": cues[1], "cue_3": cues[2],
                    "success": True, "timestamp": datetime.now().isoformat(),
                    "source": "crossgen_test"
                })
            except Exception as e:
                print(f"  Error {img_id}/{assume}: {e}")
        if (i+1) % 50 == 0:
            torch.cuda.empty_cache()
            rate = (i+1) / ((time.time()-t0)/60)
            print(f"  [Qwen] {i+1}/{len(image_paths)} | {rate:.1f} img/min")

    del model, processor
    torch.cuda.empty_cache()
    print(f"  [Qwen] Done: {len([r for r in results if r['model']=='Qwen2.5-VL-7B'])} entries")


def main():
    print("="*60)
    print("FAST VLM RUNNER - Cross-Generator Test Set")
    print(f"Input: {args.input_dir}")
    print(f"Label: {args.label}")
    print(f"Models: {args.models}")
    print("="*60)

    image_paths = collect_images(args.input_dir, args.max_images)
    print(f"Found {len(image_paths)} images")
    if not image_paths:
        print("No images found!")
        sys.exit(1)

    results = []
    if os.path.exists(args.output):
        with open(args.output) as f:
            data = json.load(f)
            results = data if isinstance(data, list) else data.get("results", [])
        print(f"Loaded {len(results)} existing entries")

    if "internvl" in args.models:
        run_internvl(image_paths, args.label, results)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"results": results}, f)

    if "qwen" in args.models:
        run_qwen(image_paths, args.label, results)
        with open(args.output, "w") as f:
            json.dump({"results": results}, f)

    print(f"\nTotal entries: {len(results)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
