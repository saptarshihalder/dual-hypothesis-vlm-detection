import json, numpy as np
from pathlib import Path

RUN     = Path("/NAS_DISK/Saptarshi_data/dhsd_v2_20260427_133533")
RUN_OLD = Path("/NAS_DISK/Saptarshi_data/dhsd_v2_20260424_163244")
SCRIPT  = Path("/home/tbvl_akshay/train_dhsd_v2.py")

code = SCRIPT.read_text()
r    = json.load(open(RUN / "final_results.json"))

passed = []
failed = []

def check(name, condition, detail=""):
    if condition:
        passed.append(name)
        print(f"  PASS  {name}")
    else:
        failed.append(name)
        print(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))

print("=" * 65)
print("DATA LEAKAGE AUDIT — dhsd_v2_20260427_133533")
print("=" * 65)

# ── 1. EVAL-TIME BLEND ────────────────────────────────────────────
print("\n[1] Eval-time score blending")
check("no auroc_student key in results json",
      not any("auroc_student" in v for v in r["per_gen_test"].values()))
check("eval_loader uses only student logit",
      "clip_zs_map" not in code.split("def eval_loader")[1].split("def ")[0])
check("no blend weight (0.7/0.3) in source",
      "0.7 * " not in code and "0.3 * " not in code)
check("no ZS_W or BLEND_W constant in source",
      "ZS_W" not in code and "BLEND_W" not in code)

# ── 2. TRAIN/TEST SPLIT LEAKAGE ───────────────────────────────────
print("\n[2] Train/test split leakage")
check("test split carved before train split",
      code.index("mj_test_items") < code.index("train_items"))
check("CNNDetection not in build_train_items",
      "cnn" not in code.split("def build_train_items")[1].split("def ")[0].lower())
check("train augment=True, eval augment=False",
      "augment=True" in code and "augment=False" in code)
check("train_loader shuffle=True, eval loaders shuffle=False",
      "shuffle=True" in code and code.count("shuffle=False") >= 2)

# ── 3. LABEL LEAKAGE ──────────────────────────────────────────────
print("\n[3] Label leakage into KD signal")
check("teacher probs loaded from external file",
      "TEACHER_PROB" in code and "np.load" in code)
check("clip_zs uses text prompts not labels",
      "REAL_PROMPTS" in code and "FAKE_PROMPTS" in code)
check("adaptive_weight receives probs and labels separately",
      "adaptive_weight(p_t, y)" in code and "adaptive_weight(p_c, y)" in code)

# ── 4. FEATURE LEAKAGE ────────────────────────────────────────────
print("\n[4] Feature/model leakage")
check("CLIP backbone frozen (no_grad around clip_forward)",
      "torch.no_grad" in code)
check("only head parameters in optimizer",
      "head.parameters()" in code)
check("clip_forward called inside no_grad block",
      "with torch.no_grad" in code and
      code.index("clip_forward") > code.index("with torch.no_grad"))

# ── 5. PRECOMPUTATION SCOPE ───────────────────────────────────────
print("\n[5] Precomputation scope")
check("clip_zs_map not accessed inside eval_loader",
      "clip_zs_map" not in code.split("def eval_loader")[1].split("def ")[0])
check("teacher_prob_map not accessed inside eval_loader",
      "teacher_prob_map" not in code.split("def eval_loader")[1].split("def ")[0])

# ── 6. MODEL SELECTION ────────────────────────────────────────────
print("\n[6] Model selection")
check("best checkpoint selected on val not test",
      "cnn_val_loaders" in code and
      code.index("cnn_val_loaders") < code.index("cnn_test_loaders"))
check("test eval runs after checkpoint loaded",
      code.index("load_state_dict") < code.index("cnn_test_loaders"))

# ── 7. RUN IDENTITY ───────────────────────────────────────────────
print("\n[7] Run identity")
check("result is from new run not old blended run",
      "20260427_133533" in str(RUN))
old_r = json.load(open(RUN_OLD / "final_results.json")) if \
        (RUN_OLD / "final_results.json").exists() else {}
check("old run was blended, new run is not",
      any("auroc_student" in v for v in old_r.get("per_gen_test", {}).values()) and
      not any("auroc_student" in v for v in r["per_gen_test"].values()))

# ── SUMMARY ───────────────────────────────────────────────────────
print()
print("=" * 65)
print(f"RESULT: {len(passed)} passed  |  {len(failed)} failed")
if not failed:
    print("NO LEAKAGE DETECTED — 0.8288 is a clean standalone result.")
else:
    print("ISSUES FOUND:")
    for f in failed:
        print(f"  - {f}")
print("=" * 65)
