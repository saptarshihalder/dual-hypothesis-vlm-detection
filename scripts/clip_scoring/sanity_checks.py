"""
=============================================================
Sanity Checks for XAI Methods (Adebayo et al., NeurIPS 2018)
=============================================================
Implements two sanity checks to verify that XAI explanations
are truly faithful to the model and data, rather than acting
as trivial edge detectors.

Tests implemented:
  1. Model Parameter Randomization Test (Cascading)
     - Progressively randomize layers from top (output) to bottom (input)
     - Faithful XAI maps should degrade as model becomes random
     - Metric: SSIM and Spearman rank correlation vs. original maps

  2. Data Randomization Test
     - Train/evaluate with randomly shuffled labels
     - XAI maps should differ from original if method is data-dependent
     - Metric: SSIM and Spearman rank correlation vs. original maps

Reference:
  Adebayo et al., "Sanity Checks for Saliency Maps", NeurIPS 2018
  https://arxiv.org/abs/1810.03292

XAI Methods tested:
  - DF-GradCAM (Decoder-Fused Grad-CAM)
  - Integrated Gradients (IG)
=============================================================
"""
import sys
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import os
import copy
import warnings
import time
import gc
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim

from brats import get_datasets
from networks.models.UNet.model import UNet3D
from tqdm import tqdm

# ============ CONFIG ============
BASE = "/home/tbvl_akshay/[REMOVED]/[REMOVED]"
WEIGHTS = f"{BASE}/unet3d_fixed/best-model/best-model.pkl"
DATASET = f"{BASE}/dataset"
OUT_DIR = f"{BASE}/test_results/faithfulness_eval_unet3d"
os.makedirs(OUT_DIR, exist_ok=True)

N_SUBJECTS = 20      # Number of test subjects (subset for efficiency)
IG_STEPS = 20        # Reduced IG steps for sanity checks (speed)
USE_FP16 = True      # Mixed precision for IG

print("=" * 70)
print("  Sanity Checks for XAI Methods (Adebayo et al., NeurIPS 2018)")
print("  Test 1: Model Parameter Randomization (Cascading)")
print("  Test 2: Data Randomization (Label Shuffling)")
print("  Methods: DF-GradCAM, Integrated Gradients")
print(f"  Subjects: {N_SUBJECTS} | IG steps: {IG_STEPS}")
print("=" * 70)


# ============ DF-GradCAM (from xai_df_gradcam_unet3d.py) ============
class DFGradCAM:
    """Decoder-Fused Grad-CAM adapted for 3D UNet.
    Ref: Selvaraju et al., ICCV 2017 + M3d-Cam (MECLabTUDA)
    """

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.layer_data = {}

        self.hooked_layers = {
            'encoder': model.bottleNeck,
            'dec_0':   model.s_block3,
            'dec_1':   model.s_block2,
        }
        self.all_layer_names = ['encoder', 'dec_0', 'dec_1', 'dec_2']

        for name in self.all_layer_names:
            self.layer_data[name] = {'act': None, 'grad': None}

        for name, layer in self.hooked_layers.items():
            def make_fwd_hook(n):
                def hook(module, inp, out):
                    if isinstance(out, tuple):
                        self.layer_data[n]['act'] = out[1].detach()
                    else:
                        self.layer_data[n]['act'] = out.detach()
                return hook

            def make_bwd_hook(n):
                def hook(module, grad_in, grad_out):
                    g = grad_out[0]
                    if isinstance(g, tuple):
                        g = g[0]
                    if g is not None:
                        self.layer_data[n]['grad'] = g.detach()
                return hook

            self.hooks.append(layer.register_forward_hook(make_fwd_hook(name)))
            self.hooks.append(layer.register_full_backward_hook(make_bwd_hook(name)))

        # dec_2: Hook s_block1.conv3's INPUT (= 64ch post-relu features).
        # Fix: conv2 captured pre-BN/pre-ReLU activations → blank heatmaps.
        # conv3's forward_pre_hook gives post-relu 64ch features.
        def _dec2_fwd_pre_hook(module, inp):
            self.layer_data['dec_2']['act'] = inp[0].detach()

        def _dec2_bwd_hook(module, grad_in, grad_out):
            if grad_in[0] is not None:
                self.layer_data['dec_2']['grad'] = grad_in[0].detach()

        self.hooks.append(model.s_block1.conv3.register_forward_pre_hook(_dec2_fwd_pre_hook))
        self.hooks.append(model.s_block1.conv3.register_full_backward_hook(_dec2_bwd_hook))

    def _compute_cam_for_layer(self, name, target_size):
        act = self.layer_data[name]['act']
        grad = self.layer_data[name]['grad']

        if act is None or grad is None:
            return np.zeros(target_size)

        weights = grad.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        if cam.shape[2:] != target_size:
            cam = F.interpolate(cam, size=target_size, mode='trilinear',
                                align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def __call__(self, input_tensor, target_channel, gt_mask=None):
        self.model.zero_grad()
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        target_size = tuple(input_tensor.shape[2:])

        out_ch = torch.sigmoid(output[0, target_channel])
        pred_mask = (out_ch.detach() > 0.5).float()
        if pred_mask.sum() > 0:
            loss = (out_ch * pred_mask).sum() / pred_mask.sum()
        else:
            loss = out_ch.mean()

        loss.backward(retain_graph=True)

        cams = {}
        for name in self.all_layer_names:
            cams[name] = self._compute_cam_for_layer(name, target_size)

        encoder_cam = cams['encoder']
        decoder_cam = cams['dec_2']

        # DF-Avg (alpha=0.3)
        alpha = 0.3
        fusion_avg = alpha * encoder_cam + (1 - alpha) * decoder_cam
        if fusion_avg.max() > 0:
            fusion_avg = (fusion_avg - fusion_avg.min()) / \
                         (fusion_avg.max() - fusion_avg.min() + 1e-8)

        return {
            'gradcam': fusion_avg,  # Use DF-Avg as the GradCAM representative
        }

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


# ============ Integrated Gradients (from xai_ig_unet3d.py) ============
class IntegratedGradients3D:
    """Memory-efficient Integrated Gradients for 3D segmentation.
    Ref: Sundararajan et al., ICML 2017
    """

    def __init__(self, model, n_steps=20, use_fp16=True):
        self.model = model
        self.n_steps = n_steps
        self.use_fp16 = use_fp16

    def compute(self, input_tensor, target_channel, gt_mask=None):
        device = input_tensor.device
        self.model.eval()

        baseline = torch.zeros_like(input_tensor)
        delta = input_tensor - baseline

        alphas = torch.linspace(0, 1, self.n_steps + 1, device=device)
        weights = torch.ones(self.n_steps + 1, device=device)
        weights[0] = 0.5
        weights[-1] = 0.5

        total_grads = torch.zeros_like(input_tensor)

        for i in range(self.n_steps + 1):
            alpha_val = alphas[i]
            interpolated = (baseline + alpha_val * delta).detach()
            interpolated.requires_grad_(True)

            if self.use_fp16 and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = self.model(interpolated)
                    output = output.float()
            else:
                output = self.model(interpolated)

            out_ch = torch.sigmoid(output[0, target_channel])
            auto_mask = (out_ch.detach() > 0.5).float()
            if auto_mask.sum() > 0:
                score = (out_ch * auto_mask).sum() / auto_mask.sum()
            else:
                score = out_ch.mean()

            self.model.zero_grad()
            score.backward()

            if interpolated.grad is not None:
                total_grads += weights[i] * interpolated.grad.detach()

            del output, score, out_ch, interpolated
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        avg_grads = total_grads / self.n_steps
        attributions = delta.detach() * avg_grads

        attr_map = torch.abs(attributions).sum(dim=1).squeeze(0).cpu().numpy()

        if attr_map.max() > 0:
            attr_map = (attr_map - attr_map.min()) / \
                       (attr_map.max() - attr_map.min() + 1e-8)

        del total_grads, avg_grads, attributions, delta
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        return attr_map


# ============ SIMILARITY METRICS ============
def compute_ssim_3d(map_a, map_b):
    """Compute SSIM between two 3D saliency maps.
    We compute SSIM slice-by-slice and average (standard for 3D volumes).
    Ref: Wang et al., "Image Quality Assessment", IEEE TIP 2004
    """
    assert map_a.shape == map_b.shape, \
        f"Shape mismatch: {map_a.shape} vs {map_b.shape}"

    # Normalize both to [0, 1]
    def norm01(x):
        if x.max() > x.min():
            return (x - x.min()) / (x.max() - x.min() + 1e-8)
        return np.zeros_like(x)

    a = norm01(map_a.astype(np.float64))
    b = norm01(map_b.astype(np.float64))

    ssim_vals = []
    for s in range(a.shape[0]):
        slice_a = a[s]
        slice_b = b[s]
        # Skip empty slices
        if slice_a.max() == 0 and slice_b.max() == 0:
            ssim_vals.append(1.0)
            continue
        val = ssim(slice_a, slice_b, data_range=1.0)
        ssim_vals.append(val)

    return float(np.mean(ssim_vals))


def compute_spearman_3d(map_a, map_b):
    """Compute Spearman rank correlation between two flattened saliency maps.
    Ref: Adebayo et al. NeurIPS 2018 use rank correlation as similarity metric.
    """
    a_flat = map_a.flatten().astype(np.float64)
    b_flat = map_b.flatten().astype(np.float64)

    # If either map is constant, correlation is undefined -> return 0
    if np.std(a_flat) < 1e-10 or np.std(b_flat) < 1e-10:
        return 0.0

    # Subsample for efficiency (128^3 = ~2M voxels)
    n = len(a_flat)
    if n > 100000:
        idx = np.random.RandomState(42).choice(n, 100000, replace=False)
        a_flat = a_flat[idx]
        b_flat = b_flat[idx]

    corr, _ = spearmanr(a_flat, b_flat)
    return float(corr) if not np.isnan(corr) else 0.0


# ============ HELPER: Compute XAI maps for a given model ============
def compute_xai_maps(model, input_tensor, target_channel, device):
    """Compute GradCAM and IG maps for a single input and channel.
    Returns dict with 'gradcam' and 'ig' keys.
    """
    maps = {}

    # GradCAM
    try:
        gcam = DFGradCAM(model)
        cam_result = gcam(input_tensor, target_channel)
        maps['gradcam'] = cam_result['gradcam']
        gcam.remove_hooks()
    except Exception as e:
        print(f"    GradCAM error: {e}")
        maps['gradcam'] = np.zeros(tuple(input_tensor.shape[2:]))

    # Integrated Gradients
    try:
        ig = IntegratedGradients3D(model, n_steps=IG_STEPS, use_fp16=USE_FP16)
        maps['ig'] = ig.compute(input_tensor, target_channel)
    except Exception as e:
        print(f"    IG error: {e}")
        maps['ig'] = np.zeros(tuple(input_tensor.shape[2:]))

    return maps


# ============ LOAD MODEL ============
print("\n[1/5] Loading 3D UNet (fixed)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet3D(in_channels=4, num_classes=3).to(device)

state = torch.load(WEIGHTS, map_location=device, weights_only=True)
if isinstance(state, tuple):
    state = state[0]
if isinstance(state, dict) and 'state_dict' in state:
    state = state['state_dict']
model.load_state_dict(state)
model.eval()
print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
print(f"  Device: {device}")


# ============ LOAD DATA ============
print("\n[2/5] Loading test dataset...")
test_dataset = get_datasets(
    dataset_folder=DATASET, mode="test",
    target_size=(128, 128, 128), version="brats2023"
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False,
    num_workers=4, pin_memory=True
)
print(f"  Test subjects: {len(test_dataset)} (using first {N_SUBJECTS})")


# ============ COLLECT TEST INPUTS ============
print("\n[3/5] Collecting test inputs...")
test_inputs = []
for i, data in enumerate(test_loader):
    if i >= N_SUBJECTS:
        break
    patient_id = data["patient_id"][0]
    inputs = data["image"].to(device)
    targets = data["label"]

    if "pad_list" in data:
        pl = [p.item() if torch.is_tensor(p) else p for p in data["pad_list"]]
        if any(p > 0 for p in pl):
            targets = targets[:, :,
                pl[-4]:targets.shape[2]-pl[-3] if pl[-3] > 0 else targets.shape[2],
                pl[-6]:targets.shape[3]-pl[-5] if pl[-5] > 0 else targets.shape[3],
                pl[-8]:targets.shape[4]-pl[-7] if pl[-7] > 0 else targets.shape[4]]

    test_inputs.append({
        'patient_id': patient_id,
        'inputs': inputs.detach(),
        'targets': targets.squeeze().numpy(),
    })
    print(f"  Loaded: {patient_id}")

print(f"  Collected {len(test_inputs)} subjects")


# ============ COMPUTE ORIGINAL XAI MAPS ============
print("\n[4/5] Computing original XAI maps (baseline)...")
original_maps = {}  # {patient_id: {channel: {'gradcam': ..., 'ig': ...}}}
channel_names = ['ET', 'TC', 'WT']

start_time = time.time()
for idx, sample in enumerate(tqdm(test_inputs, desc="Original XAI")):
    pid = sample['patient_id']
    inp = sample['inputs']
    original_maps[pid] = {}

    for ch in range(3):
        original_maps[pid][ch] = compute_xai_maps(model, inp, ch, device)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    if (idx + 1) % 5 == 0:
        elapsed = time.time() - start_time
        eta = elapsed / (idx + 1) * (len(test_inputs) - idx - 1)
        print(f"  [{idx+1}/{len(test_inputs)}] Elapsed: {elapsed/60:.1f}min | "
              f"ETA: {eta/60:.1f}min")

orig_time = time.time() - start_time
print(f"  Original maps computed in {orig_time/60:.1f} min")


# ================================================================
#  TEST 1: MODEL PARAMETER RANDOMIZATION (CASCADING)
# ================================================================
print("\n" + "=" * 70)
print("  TEST 1: Model Parameter Randomization (Cascading)")
print("  Ref: Adebayo et al. NeurIPS 2018, Section 3.1")
print("  Progressive randomization from top (output) to bottom (input)")
print("=" * 70)

# Get named modules with learnable parameters (in order)
# We reverse so we randomize from top (output end) to bottom (input end)
named_layers = []
for name, module in model.named_modules():
    # Only consider layers with actual parameters (Conv3d, BatchNorm, etc.)
    if hasattr(module, 'weight') and module.weight is not None:
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d,
                               nn.InstanceNorm3d, nn.BatchNorm3d, nn.Linear)):
            named_layers.append((name, module))

print(f"\n  Found {len(named_layers)} layers with learnable parameters")
# Print layer names for reference
for i, (name, mod) in enumerate(named_layers):
    n_params = sum(p.numel() for p in mod.parameters())
    print(f"    [{i:2d}] {name:<45} {type(mod).__name__:<20} "
          f"params={n_params:>8,}")

# Reverse: top (output) to bottom (input) for cascading randomization
# Ref: Adebayo et al. cascade from logit layer down
named_layers_reversed = list(reversed(named_layers))

# Select a subset of randomization levels for efficiency
# We pick ~8 evenly spaced levels plus the fully randomized state
n_total_layers = len(named_layers_reversed)
if n_total_layers <= 10:
    randomize_at = list(range(1, n_total_layers + 1))
else:
    # Select ~8-10 evenly spaced levels
    step = max(1, n_total_layers // 8)
    randomize_at = list(range(step, n_total_layers, step))
    if n_total_layers not in randomize_at:
        randomize_at.append(n_total_layers)

print(f"\n  Randomization levels: {randomize_at}")
print(f"  (out of {n_total_layers} total layers)")

# Run cascading randomization
model_rand_results = []  # List of dicts
start_time = time.time()

for n_layers_rand in tqdm(randomize_at, desc="Cascading Randomization"):
    # Deep copy the original model
    rand_model = copy.deepcopy(model)
    rand_model.eval()

    # Randomize top n_layers_rand layers (from output side)
    layers_randomized_names = []
    for j in range(n_layers_rand):
        layer_name, _ = named_layers_reversed[j]
        layers_randomized_names.append(layer_name)

        # Find the module in the copied model
        parts = layer_name.split('.')
        mod = rand_model
        for part in parts:
            mod = getattr(mod, part)

        # Reinitialize parameters with random values
        for param in mod.parameters():
            nn.init.normal_(param, mean=0.0, std=0.02)

    pct_randomized = n_layers_rand / n_total_layers * 100
    print(f"\n  Level {n_layers_rand}/{n_total_layers} "
          f"({pct_randomized:.0f}% randomized from top)")
    print(f"    Layers randomized: {layers_randomized_names[-1]} ... "
          f"(last randomized)")

    # Compute XAI maps with randomized model
    level_ssim_gc = []
    level_spearman_gc = []
    level_ssim_ig = []
    level_spearman_ig = []

    for sample in test_inputs:
        pid = sample['patient_id']
        inp = sample['inputs']

        for ch in range(3):
            rand_maps = compute_xai_maps(rand_model, inp, ch, device)
            orig_gc = original_maps[pid][ch]['gradcam']
            orig_ig = original_maps[pid][ch]['ig']
            rand_gc = rand_maps['gradcam']
            rand_ig = rand_maps['ig']

            # Ensure same shape
            min_shape = tuple(min(a, b) for a, b in
                              zip(orig_gc.shape, rand_gc.shape))
            orig_gc_c = orig_gc[:min_shape[0], :min_shape[1], :min_shape[2]]
            rand_gc_c = rand_gc[:min_shape[0], :min_shape[1], :min_shape[2]]
            orig_ig_c = orig_ig[:min_shape[0], :min_shape[1], :min_shape[2]]
            rand_ig_c = rand_ig[:min_shape[0], :min_shape[1], :min_shape[2]]

            level_ssim_gc.append(compute_ssim_3d(orig_gc_c, rand_gc_c))
            level_spearman_gc.append(compute_spearman_3d(orig_gc_c, rand_gc_c))
            level_ssim_ig.append(compute_ssim_3d(orig_ig_c, rand_ig_c))
            level_spearman_ig.append(compute_spearman_3d(orig_ig_c, rand_ig_c))

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    result = {
        'n_layers_randomized': n_layers_rand,
        'pct_randomized': pct_randomized,
        'last_layer_randomized': named_layers_reversed[n_layers_rand - 1][0],
        'gradcam_ssim_mean': float(np.mean(level_ssim_gc)),
        'gradcam_ssim_std': float(np.std(level_ssim_gc)),
        'gradcam_spearman_mean': float(np.mean(level_spearman_gc)),
        'gradcam_spearman_std': float(np.std(level_spearman_gc)),
        'ig_ssim_mean': float(np.mean(level_ssim_ig)),
        'ig_ssim_std': float(np.std(level_ssim_ig)),
        'ig_spearman_mean': float(np.mean(level_spearman_ig)),
        'ig_spearman_std': float(np.std(level_spearman_ig)),
    }
    model_rand_results.append(result)

    print(f"    GradCAM  — SSIM: {result['gradcam_ssim_mean']:.4f} +/- "
          f"{result['gradcam_ssim_std']:.4f} | "
          f"Spearman: {result['gradcam_spearman_mean']:.4f} +/- "
          f"{result['gradcam_spearman_std']:.4f}")
    print(f"    IG       — SSIM: {result['ig_ssim_mean']:.4f} +/- "
          f"{result['ig_ssim_std']:.4f} | "
          f"Spearman: {result['ig_spearman_mean']:.4f} +/- "
          f"{result['ig_spearman_std']:.4f}")

    # Cleanup
    del rand_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

rand_time = time.time() - start_time
print(f"\n  Model randomization test completed in {rand_time/60:.1f} min")


# ================================================================
#  TEST 2: DATA RANDOMIZATION (LABEL SHUFFLING)
# ================================================================
print("\n" + "=" * 70)
print("  TEST 2: Data Randomization (Label Shuffling)")
print("  Ref: Adebayo et al. NeurIPS 2018, Section 3.2")
print("  Shuffle input labels, compute XAI maps on same images")
print("=" * 70)

# For the data randomization test, we do NOT retrain the model.
# Instead, we simulate what happens when the target channel used for
# backpropagation is wrong (randomly assigned).
# Ref: Adebayo et al. compare saliency on correct vs random labels.
# For segmentation: we use a different (random) channel as the target.

data_rand_results = []
start_time = time.time()

np.random.seed(42)

# Create shuffled channel assignments for each subject
# Each subject gets a random permutation of channels
shuffled_channels = {}
for sample in test_inputs:
    pid = sample['patient_id']
    # For each original channel, pick a random different channel
    perm = np.random.permutation(3).tolist()
    # Ensure at least one channel is different
    while perm == [0, 1, 2]:
        perm = np.random.permutation(3).tolist()
    shuffled_channels[pid] = perm

print(f"\n  Computing XAI maps with shuffled target channels...")

data_ssim_gc = []
data_spearman_gc = []
data_ssim_ig = []
data_spearman_ig = []

per_subject_data_results = []

for idx, sample in enumerate(tqdm(test_inputs, desc="Data Randomization")):
    pid = sample['patient_id']
    inp = sample['inputs']
    perm = shuffled_channels[pid]

    subj_ssim_gc = []
    subj_spearman_gc = []
    subj_ssim_ig = []
    subj_spearman_ig = []

    for orig_ch in range(3):
        shuffled_ch = perm[orig_ch]

        # Compute XAI map using the WRONG (shuffled) target channel
        shuffled_maps = compute_xai_maps(model, inp, shuffled_ch, device)

        orig_gc = original_maps[pid][orig_ch]['gradcam']
        orig_ig = original_maps[pid][orig_ch]['ig']
        shuf_gc = shuffled_maps['gradcam']
        shuf_ig = shuffled_maps['ig']

        # Ensure same shape
        min_shape = tuple(min(a, b) for a, b in
                          zip(orig_gc.shape, shuf_gc.shape))
        orig_gc_c = orig_gc[:min_shape[0], :min_shape[1], :min_shape[2]]
        shuf_gc_c = shuf_gc[:min_shape[0], :min_shape[1], :min_shape[2]]
        orig_ig_c = orig_ig[:min_shape[0], :min_shape[1], :min_shape[2]]
        shuf_ig_c = shuf_ig[:min_shape[0], :min_shape[1], :min_shape[2]]

        s_gc = compute_ssim_3d(orig_gc_c, shuf_gc_c)
        r_gc = compute_spearman_3d(orig_gc_c, shuf_gc_c)
        s_ig = compute_ssim_3d(orig_ig_c, shuf_ig_c)
        r_ig = compute_spearman_3d(orig_ig_c, shuf_ig_c)

        subj_ssim_gc.append(s_gc)
        subj_spearman_gc.append(r_gc)
        subj_ssim_ig.append(s_ig)
        subj_spearman_ig.append(r_ig)

        data_ssim_gc.append(s_gc)
        data_spearman_gc.append(r_gc)
        data_ssim_ig.append(s_ig)
        data_spearman_ig.append(r_ig)

    per_subject_data_results.append({
        'patient_id': pid,
        'original_channels': '0,1,2',
        'shuffled_channels': ','.join(map(str, perm)),
        'gradcam_ssim_mean': float(np.mean(subj_ssim_gc)),
        'gradcam_spearman_mean': float(np.mean(subj_spearman_gc)),
        'ig_ssim_mean': float(np.mean(subj_ssim_ig)),
        'ig_spearman_mean': float(np.mean(subj_spearman_ig)),
    })

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

data_time = time.time() - start_time
print(f"\n  Data randomization test completed in {data_time/60:.1f} min")


# ================================================================
#  SAVE RESULTS
# ================================================================
print("\n[5/5] Saving results...")

# --- Model Randomization CSV ---
model_rand_csv = os.path.join(OUT_DIR, "sanity_checks_model_randomization.csv")
with open(model_rand_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=model_rand_results[0].keys())
    writer.writeheader()
    writer.writerows(model_rand_results)
print(f"  Model randomization: {model_rand_csv}")

# --- Data Randomization CSV ---
data_rand_csv = os.path.join(OUT_DIR, "sanity_checks_data_randomization.csv")
with open(data_rand_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=per_subject_data_results[0].keys())
    writer.writeheader()
    writer.writerows(per_subject_data_results)
print(f"  Data randomization: {data_rand_csv}")

# --- Combined Summary CSV ---
summary_csv = os.path.join(OUT_DIR, "sanity_checks_results.csv")
with open(summary_csv, 'w', newline='') as f:
    writer = csv.writer(f)

    # Header
    writer.writerow(['test_type', 'method', 'metric', 'value', 'std', 'detail'])

    # Model randomization results
    for res in model_rand_results:
        n = res['n_layers_randomized']
        pct = res['pct_randomized']
        detail = f"layers_rand={n}, pct={pct:.0f}%"
        writer.writerow(['model_randomization', 'GradCAM', 'SSIM',
                          f"{res['gradcam_ssim_mean']:.4f}",
                          f"{res['gradcam_ssim_std']:.4f}", detail])
        writer.writerow(['model_randomization', 'GradCAM', 'Spearman',
                          f"{res['gradcam_spearman_mean']:.4f}",
                          f"{res['gradcam_spearman_std']:.4f}", detail])
        writer.writerow(['model_randomization', 'IG', 'SSIM',
                          f"{res['ig_ssim_mean']:.4f}",
                          f"{res['ig_ssim_std']:.4f}", detail])
        writer.writerow(['model_randomization', 'IG', 'Spearman',
                          f"{res['ig_spearman_mean']:.4f}",
                          f"{res['ig_spearman_std']:.4f}", detail])

    # Data randomization summary
    writer.writerow(['data_randomization', 'GradCAM', 'SSIM',
                      f"{np.mean(data_ssim_gc):.4f}",
                      f"{np.std(data_ssim_gc):.4f}",
                      f"n_subjects={len(test_inputs)}"])
    writer.writerow(['data_randomization', 'GradCAM', 'Spearman',
                      f"{np.mean(data_spearman_gc):.4f}",
                      f"{np.std(data_spearman_gc):.4f}",
                      f"n_subjects={len(test_inputs)}"])
    writer.writerow(['data_randomization', 'IG', 'SSIM',
                      f"{np.mean(data_ssim_ig):.4f}",
                      f"{np.std(data_ssim_ig):.4f}",
                      f"n_subjects={len(test_inputs)}"])
    writer.writerow(['data_randomization', 'IG', 'Spearman',
                      f"{np.mean(data_spearman_ig):.4f}",
                      f"{np.std(data_spearman_ig):.4f}",
                      f"n_subjects={len(test_inputs)}"])

print(f"  Combined summary: {summary_csv}")


# ================================================================
#  PLOT: MODEL PARAMETER RANDOMIZATION CASCADE
# ================================================================
print("\n  Generating cascade plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

pcts = [r['pct_randomized'] for r in model_rand_results]
# Prepend 0% (original model = SSIM 1.0, Spearman 1.0)
pcts_plot = [0.0] + pcts

# SSIM cascade
gc_ssim = [1.0] + [r['gradcam_ssim_mean'] for r in model_rand_results]
ig_ssim = [1.0] + [r['ig_ssim_mean'] for r in model_rand_results]
gc_ssim_err = [0.0] + [r['gradcam_ssim_std'] for r in model_rand_results]
ig_ssim_err = [0.0] + [r['ig_ssim_std'] for r in model_rand_results]

axes[0].errorbar(pcts_plot, gc_ssim, yerr=gc_ssim_err,
                 marker='o', linewidth=2, capsize=4, label='DF-GradCAM',
                 color='#e74c3c')
axes[0].errorbar(pcts_plot, ig_ssim, yerr=ig_ssim_err,
                 marker='s', linewidth=2, capsize=4,
                 label='Integrated Gradients', color='#3498db')
axes[0].set_xlabel('Layers Randomized (%)', fontsize=13)
axes[0].set_ylabel('SSIM with Original Map', fontsize=13)
axes[0].set_title('Model Randomization: SSIM Cascade', fontsize=14,
                   fontweight='bold')
axes[0].set_ylim(-0.05, 1.05)
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

# Spearman cascade
gc_spear = [1.0] + [r['gradcam_spearman_mean'] for r in model_rand_results]
ig_spear = [1.0] + [r['ig_spearman_mean'] for r in model_rand_results]
gc_spear_err = [0.0] + [r['gradcam_spearman_std'] for r in model_rand_results]
ig_spear_err = [0.0] + [r['ig_spearman_std'] for r in model_rand_results]

axes[1].errorbar(pcts_plot, gc_spear, yerr=gc_spear_err,
                 marker='o', linewidth=2, capsize=4, label='DF-GradCAM',
                 color='#e74c3c')
axes[1].errorbar(pcts_plot, ig_spear, yerr=ig_spear_err,
                 marker='s', linewidth=2, capsize=4,
                 label='Integrated Gradients', color='#3498db')
axes[1].set_xlabel('Layers Randomized (%)', fontsize=13)
axes[1].set_ylabel('Spearman Correlation with Original Map', fontsize=13)
axes[1].set_title('Model Randomization: Spearman Cascade', fontsize=14,
                   fontweight='bold')
axes[1].set_ylim(-0.15, 1.05)
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
cascade_plot_path = os.path.join(OUT_DIR, "sanity_check_model_randomization_cascade.png")
plt.savefig(cascade_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Cascade plot: {cascade_plot_path}")


# ================================================================
#  PLOT: DATA RANDOMIZATION BAR CHART
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

methods = ['DF-GradCAM', 'Integrated Gradients']
ssim_means = [np.mean(data_ssim_gc), np.mean(data_ssim_ig)]
ssim_stds = [np.std(data_ssim_gc), np.std(data_ssim_ig)]
spear_means = [np.mean(data_spearman_gc), np.mean(data_spearman_ig)]
spear_stds = [np.std(data_spearman_gc), np.std(data_spearman_ig)]
colors = ['#e74c3c', '#3498db']

bars1 = axes[0].bar(methods, ssim_means, yerr=ssim_stds, capsize=6,
                     color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
axes[0].set_ylabel('SSIM with Original Map', fontsize=13)
axes[0].set_title('Data Randomization: SSIM', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, 1.1)
axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
                label='Identical (unfaithful)')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, ssim_means):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=12,
                 fontweight='bold')

bars2 = axes[1].bar(methods, spear_means, yerr=spear_stds, capsize=6,
                     color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
axes[1].set_ylabel('Spearman Correlation with Original Map', fontsize=13)
axes[1].set_title('Data Randomization: Spearman', fontsize=14,
                   fontweight='bold')
axes[1].set_ylim(-0.2, 1.1)
axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
                label='Identical (unfaithful)')
axes[1].axhline(y=0.0, color='gray', linestyle=':', alpha=0.5,
                label='Uncorrelated')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, spear_means):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=12,
                 fontweight='bold')

plt.tight_layout()
data_plot_path = os.path.join(OUT_DIR, "sanity_check_data_randomization.png")
plt.savefig(data_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Data randomization plot: {data_plot_path}")


# ================================================================
#  PRINT FORMATTED SUMMARY
# ================================================================
total_time = orig_time + rand_time + data_time

print("\n" + "=" * 70)
print("  SANITY CHECKS SUMMARY")
print("  Ref: Adebayo et al., NeurIPS 2018")
print(f"  Subjects: {len(test_inputs)} | Total time: {total_time/60:.1f} min")
print("=" * 70)

print("\n  --- TEST 1: Model Parameter Randomization (Cascading) ---")
print("  Expected: SSIM and Spearman should DECREASE as more layers are randomized")
print("  If they stay high -> method is NOT faithful (edge detector)")
print()
print(f"  {'% Randomized':<15} {'GradCAM SSIM':>14} {'GradCAM Spear':>14} "
      f"{'IG SSIM':>14} {'IG Spear':>14}")
print(f"  {'-'*75}")
print(f"  {'0% (original)':<15} {'1.0000':>14} {'1.0000':>14} "
      f"{'1.0000':>14} {'1.0000':>14}")

for r in model_rand_results:
    pct_str = f"{r['pct_randomized']:.0f}%"
    print(f"  {pct_str:<15} "
          f"{r['gradcam_ssim_mean']:>14.4f} "
          f"{r['gradcam_spearman_mean']:>14.4f} "
          f"{r['ig_ssim_mean']:>14.4f} "
          f"{r['ig_spearman_mean']:>14.4f}")

# Assess pass/fail
final_gc_ssim = model_rand_results[-1]['gradcam_ssim_mean']
final_gc_spear = model_rand_results[-1]['gradcam_spearman_mean']
final_ig_ssim = model_rand_results[-1]['ig_ssim_mean']
final_ig_spear = model_rand_results[-1]['ig_spearman_mean']

print()
print(f"  At 100% randomization:")
gc_pass = final_gc_ssim < 0.7 or final_gc_spear < 0.5
ig_pass = final_ig_ssim < 0.7 or final_ig_spear < 0.5
print(f"    GradCAM  — SSIM={final_gc_ssim:.4f}, Spearman={final_gc_spear:.4f} "
      f"-> {'PASS (faithful)' if gc_pass else 'FAIL (may be edge detector)'}")
print(f"    IG       — SSIM={final_ig_ssim:.4f}, Spearman={final_ig_spear:.4f} "
      f"-> {'PASS (faithful)' if ig_pass else 'FAIL (may be edge detector)'}")

print("\n  --- TEST 2: Data Randomization (Label Shuffling) ---")
print("  Expected: SSIM and Spearman should be LOW (maps differ with wrong labels)")
print("  If they stay high -> method is NOT data-dependent")
print()
print(f"  {'Method':<25} {'SSIM':>14} {'Spearman':>14} {'Verdict':>20}")
print(f"  {'-'*75}")

gc_data_ssim = np.mean(data_ssim_gc)
gc_data_spear = np.mean(data_spearman_gc)
ig_data_ssim = np.mean(data_ssim_ig)
ig_data_spear = np.mean(data_spearman_ig)

gc_data_pass = gc_data_ssim < 0.8 or gc_data_spear < 0.6
ig_data_pass = ig_data_ssim < 0.8 or ig_data_spear < 0.6

print(f"  {'DF-GradCAM':<25} {gc_data_ssim:>14.4f} {gc_data_spear:>14.4f} "
      f"{'PASS (data-dep)' if gc_data_pass else 'FAIL (not data-dep)':>20}")
print(f"  {'Integrated Gradients':<25} {ig_data_ssim:>14.4f} {ig_data_spear:>14.4f} "
      f"{'PASS (data-dep)' if ig_data_pass else 'FAIL (not data-dep)':>20}")

print("\n  --- OUTPUT FILES ---")
print(f"  Combined CSV:     {summary_csv}")
print(f"  Model Rand CSV:   {model_rand_csv}")
print(f"  Data Rand CSV:    {data_rand_csv}")
print(f"  Cascade Plot:     {cascade_plot_path}")
print(f"  Data Rand Plot:   {data_plot_path}")

print(f"\nDone! Total time: {total_time/60:.1f} min")
