#!/usr/bin/env python3
"""
Compare Original Alpamayo vs Fine-tuned LoRA on both Off-road and Original data.
Visualizes trajectory predictions side-by-side.
"""

import sys
import copy
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/byounggun/alpamayo/src')

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from peft import PeftModel

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

'''
source ~/ar1_venv/bin/activate
cd /home/byounggun/alpamayo/src
python -m alpamayo_r1.alignment.compare_models
'''

# Configuration
NUM_SAMPLES = 10
LORA_CHECKPOINT = "/home/byounggun/alpamayo/outputs/alpamayo_lora_rellis3d/checkpoint-900"
RELLIS3D_ROOT = Path("/home/byounggun/alpamayo/Rellis-3D")
FINETUNE_DATA = Path("/home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset/finetune_data.jsonl")
OUTPUT_DIR = Path("/home/byounggun/alpamayo/outputs/comparison_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

NUM_FRAMES = 4

print("="*60)
print("MODEL COMPARISON: Original vs Fine-tuned Alpamayo")
print("="*60)

# ==============================================================================
# Helper Functions
# ==============================================================================
def create_ego_history(device):
    """Create realistic straight-line ego history."""
    num_history = 16
    dt = 0.1
    speed = 5.0
    
    times = np.arange(-num_history + 1, 1) * dt
    positions = np.zeros((num_history, 3))
    positions[:, 0] = times * speed
    
    ego_history_xyz = torch.from_numpy(positions).float().unsqueeze(0).unsqueeze(0).to(device)
    ego_history_rot = torch.eye(3).unsqueeze(0).repeat(num_history, 1, 1)
    ego_history_rot = ego_history_rot.unsqueeze(0).unsqueeze(0).to(device)
    
    return ego_history_xyz, ego_history_rot

def run_inference(model, processor, frame_paths, device="cuda"):
    """Run inference on a set of frames."""
    frames = []
    for fp in frame_paths:
        img = Image.open(fp).convert("RGB")
        img_np = np.array(img)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        frames.append(img_t)
    
    frames_tensor = torch.stack(frames, dim=0)
    messages = helper.create_message(frames_tensor)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    ego_history_xyz, ego_history_rot = create_ego_history(device)
    
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    model_inputs = helper.to_device(model_inputs, device)
    
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=copy.deepcopy(model_inputs),
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=True,
            )
    
    trajectory = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]
    reasoning = extra["cot"][0, 0, 0] if extra.get("cot") is not None else ""
    
    return trajectory, reasoning

def visualize_comparison(image_path, traj_original, traj_finetuned, traj_gt=None, 
                         reasoning_orig="", reasoning_ft="", save_path=None, title=""):
    """Create side-by-side comparison visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Load and show image
    img = Image.open(image_path)
    axes[0].imshow(img)
    axes[0].set_title(f"Input Image\n{title}", fontsize=10)
    axes[0].axis('off')
    
    # Plot trajectories comparison
    ax = axes[1]
    ax.plot(traj_original[:, 1], traj_original[:, 0], 'b-o', label='Original Alpamayo', linewidth=2, markersize=4)
    ax.plot(traj_finetuned[:, 1], traj_finetuned[:, 0], 'r-o', label='Fine-tuned (LoRA)', linewidth=2, markersize=4)
    if traj_gt is not None:
        ax.plot(traj_gt[:, 1], traj_gt[:, 0], 'g--', label='GT/Corrected', linewidth=2, alpha=0.7)
    ax.scatter([0], [0], c='black', s=100, marker='^', zorder=5, label='Ego')
    ax.set_xlabel('Y (lateral)')
    ax.set_ylabel('X (forward)')
    ax.set_title('Trajectory Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-2, 25)
    
    # Reasoning comparison
    ax = axes[2]
    ax.axis('off')
    
    # Handle empty reasoning
    orig_text = reasoning_orig if reasoning_orig else "[No reasoning output]"
    ft_text = reasoning_ft if reasoning_ft else "[No reasoning output]"
    
    # Truncate for display
    orig_display = orig_text[:500] + "..." if len(orig_text) > 500 else orig_text
    ft_display = ft_text[:500] + "..." if len(ft_text) > 500 else ft_text
    
    text = f"== Original Alpamayo ==\n{orig_display}\n\n== Fine-tuned ==\n{ft_display}"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=7, 
            verticalalignment='top', fontfamily='monospace', wrap=True)
    ax.set_title('Reasoning Comparison')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ==============================================================================
# Load Test Samples First
# ==============================================================================
print("\n[1/5] Loading test samples...")

# Load from finetune data (off-road samples)
with open(FINETUNE_DATA, "r") as f:
    all_samples = [json.loads(line) for line in f]

# Sample evenly
sample_indices = np.linspace(0, len(all_samples)-1, NUM_SAMPLES, dtype=int)
test_samples = [all_samples[i] for i in sample_indices]

print(f"✓ Selected {len(test_samples)} test samples")

# ==============================================================================
# Phase 1: Run Original Model
# ==============================================================================
print("\n[2/5] Loading Original Alpamayo model...")
original_model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(original_model.tokenizer)
print("✓ Original model loaded!")

print("\n[3/5] Running inference with Original model...")
original_results = {}
for i, sample in enumerate(test_samples):
    print(f"  Processing sample {i+1}/{len(test_samples)}...", end=" ")
    frame_paths = sample['frame_paths']
    
    if not all(Path(fp).exists() for fp in frame_paths):
        print("⚠ Skipping - missing files")
        continue
    
    try:
        traj, reason = run_inference(original_model, processor, frame_paths)
        original_results[i] = {"trajectory": traj, "reasoning": reason}
        print("✓")
    except Exception as e:
        print(f"✗ Error: {str(e)[:30]}")

# Clear GPU memory
del original_model
torch.cuda.empty_cache()
import gc
gc.collect()
print("✓ GPU memory cleared")

# ==============================================================================
# Phase 2: Run Fine-tuned Model
# ==============================================================================
print("\n[4/5] Loading Fine-tuned LoRA model...")
finetuned_base = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16)

# Load LoRA adapter (keep as PeftModel - like verify_finetuned.py)
print(f"  Loading adapter from: {LORA_CHECKPOINT}")
finetuned_base.vlm = PeftModel.from_pretrained(finetuned_base.vlm, LORA_CHECKPOINT)
print(f"  ✓ LoRA adapter loaded!")

# Move to GPU and set eval mode
finetuned_model = finetuned_base.to("cuda")
finetuned_model.eval()
processor = helper.get_processor(finetuned_model.tokenizer)
print("✓ Fine-tuned model ready!")

print("\nRunning inference with Fine-tuned model...")
finetuned_results = {}
for i, sample in enumerate(test_samples):
    print(f"  Processing sample {i+1}/{len(test_samples)}...", end=" ")
    frame_paths = sample['frame_paths']
    
    if not all(Path(fp).exists() for fp in frame_paths):
        print("⚠ Skipping - missing files")
        continue
    
    try:
        traj, reason = run_inference(finetuned_model, processor, frame_paths)
        finetuned_results[i] = {"trajectory": traj, "reasoning": reason}
        # Debug: show reasoning length
        print(f"✓ (reasoning: {len(reason)} chars)")
    except Exception as e:
        print(f"✗ Error: {str(e)[:30]}")

# Clear GPU memory
del finetuned_model
torch.cuda.empty_cache()
gc.collect()

# ==============================================================================
# Phase 3: Generate Visualizations
# ==============================================================================
print("\n[5/5] Generating visualizations...")

results = []
for i, sample in enumerate(test_samples):
    if i not in original_results or i not in finetuned_results:
        continue
    
    frame_paths = sample['frame_paths']
    traj_gt = np.array(sample.get('trajectory', [])) if sample.get('trajectory') else None
    
    save_path = OUTPUT_DIR / f"comparison_{i:02d}_{sample['folder']}.png"
    visualize_comparison(
        image_path=frame_paths[-1],
        traj_original=original_results[i]["trajectory"],
        traj_finetuned=finetuned_results[i]["trajectory"],
        traj_gt=traj_gt,
        reasoning_orig=original_results[i]["reasoning"],
        reasoning_ft=finetuned_results[i]["reasoning"],
        save_path=save_path,
        title=f"Sample {i+1} - {sample['folder']} - Source: {sample.get('source', 'unknown')}"
    )
    
    results.append({
        "sample_id": i,
        "folder": sample['folder'],
        "source": sample.get('source', 'unknown'),
        "violated": sample.get('violated', False),
        "traj_original": original_results[i]["trajectory"].tolist(),
        "traj_finetuned": finetuned_results[i]["trajectory"].tolist(),
        "traj_gt": traj_gt.tolist() if traj_gt is not None else None,
    })
    
    print(f"  ✓ Saved: {save_path.name}")

# Save results JSON
results_path = OUTPUT_DIR / "comparison_results.json"
with open(results_path, "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "lora_checkpoint": LORA_CHECKPOINT,
        "num_samples": len(results),
        "results": results,
    }, f, indent=2)

print("\n" + "="*60)
print("COMPARISON COMPLETE!")
print("="*60)
print(f"Saved {len(results)} comparisons to: {OUTPUT_DIR}")
print(f"  - comparison_XX_*.png (visualizations)")
print(f"  - comparison_results.json (data)")
