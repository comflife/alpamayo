#!/usr/bin/env python3
"""
Improved Model Comparison Visualization (Grid Style)

Creates a grid layout comparing Original vs Fine-tuned Alpamayo models
with projected trajectories on images + BEV plots.

Similar to viz_training_data.py but for model comparison.
"""

import os
import sys
import copy
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from peft import PeftModel, LoraConfig, get_peft_model

sys.path.insert(0, '/home/byounggun/alpamayo/src')

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# Configuration
CHECKPOINT_PATH = "/home/byounggun/alpamayo/outputs/alpamayo_full_finetuned"
DATA_PATH = "/home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset/finetune_data.jsonl"
OUTPUT_DIR = "/home/byounggun/alpamayo/outputs/comparison_visualizations_v2"
DEVICE_ORIGINAL = "cuda:0"
DEVICE_FINETUNED = "cuda:1"
NUM_SAMPLES = 20  # Number of samples to visualize

# Camera intrinsics (Rellis-3D style)
FX, FY = 2813.64, 2808.33
CX, CY = 969.29, 624.05

os.makedirs(OUTPUT_DIR, exist_ok=True)


def project_to_image(trajectory_xy, img_shape):
    """Project BEV trajectory to image coordinates."""
    H, W = img_shape[:2]
    points = []
    for x_fwd, y_lat in trajectory_xy:
        if x_fwd <= 1.0:
            continue
        u = CX - FX * y_lat / x_fwd
        v = CY + FY * 1.5 / x_fwd  # 1.5m camera height
        if 0 <= u < W and 0 <= v < H:
            points.append((int(u), int(v)))
    return points


def create_ego_history(device):
    """Create ego history for inference."""
    num_history = 16
    dt = 0.1
    speed = 5.0
    times = np.arange(-num_history + 1, 1) * dt
    positions = np.zeros((num_history, 3))
    positions[:, 0] = times * speed
    
    ego_history_xyz = torch.from_numpy(positions).float().unsqueeze(0).unsqueeze(0).to(device)
    ego_history_rot = torch.eye(3).unsqueeze(0).repeat(num_history, 1, 1).unsqueeze(0).unsqueeze(0).to(device)
    
    return ego_history_xyz, ego_history_rot


def load_original_model():
    """Load original Alpamayo model."""
    print(f"\nLoading ORIGINAL model on {DEVICE_ORIGINAL}...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
    ).to(DEVICE_ORIGINAL)
    model.eval()
    print("✓ Original model loaded")
    return model


def load_finetuned_model():
    """Load fine-tuned Alpamayo model."""
    print(f"\nLoading FINE-TUNED model on {DEVICE_FINETUNED}...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
    ).to(DEVICE_FINETUNED)
    
    # Load VLM LoRA
    vlm_lora_path = os.path.join(CHECKPOINT_PATH, "final", "vlm_lora")
    if os.path.exists(vlm_lora_path):
        model.vlm = PeftModel.from_pretrained(model.vlm, vlm_lora_path)
        print("  ✓ VLM LoRA loaded")
    
    # Load Expert + Diffusion
    checkpoint_path = os.path.join(CHECKPOINT_PATH, "final", "expert_diffusion.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE_FINETUNED)
        
        if "expert_lora" in checkpoint:
            expert_lora_config = LoraConfig(
                r=8, lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05, bias="none", task_type="FEATURE_EXTRACTION",
            )
            model.expert = get_peft_model(model.expert, expert_lora_config)
            model.expert.load_state_dict(checkpoint["expert_lora"], strict=False)
            print("  ✓ Expert LoRA loaded")
        
        if "action_in_proj" in checkpoint:
            model.action_in_proj.load_state_dict(checkpoint["action_in_proj"])
        if "action_out_proj" in checkpoint:
            model.action_out_proj.load_state_dict(checkpoint["action_out_proj"])
    
    model.eval()
    print("✓ Fine-tuned model loaded")
    return model


def run_inference(model, processor, frame_paths, device):
    """Run model inference."""
    frames = []
    for fp in frame_paths:
        if os.path.exists(fp):
            img = Image.open(fp).convert("RGB")
            img_np = np.array(img)
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            frames.append(img_t)
    
    if not frames:
        return None, None
    
    frames_tensor = torch.stack(frames, dim=0)
    messages = helper.create_message(frames_tensor)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
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
                top_p=0.98, temperature=0.6, num_traj_samples=1,
                max_generation_length=256, return_extra=True,
            )
    
    # Extract trajectory: (B, S, num_samples, T, 3) -> (T, 2)
    trajectory = pred_xyz[0, 0, 0, :, :2].cpu().numpy()
    reasoning = extra["cot"][0, 0, 0] if extra.get("cot") is not None else "No reasoning"
    
    return trajectory, reasoning


def main():
    print("=" * 60)
    print("ALPAMAYO MODEL COMPARISON (Grid Visualization)")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading dataset from {DATA_PATH}...")
    with open(DATA_PATH, "r") as f:
        samples = [json.loads(line) for line in f]
    
    # Select diverse samples
    total_samples = len(samples)
    indices = np.linspace(0, total_samples - 1, NUM_SAMPLES, dtype=int)
    selected_samples = [samples[i] for i in indices]
    print(f"Selected {NUM_SAMPLES} samples from {total_samples} total")
    
    # Load models
    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)
    
    original_model = load_original_model()
    processor_orig = helper.get_processor(original_model.tokenizer)
    
    finetuned_model = load_finetuned_model()
    processor_ft = helper.get_processor(finetuned_model.tokenizer)
    
    # Run inference on all samples
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE")
    print("=" * 60)
    
    results = []
    
    for idx, sample in enumerate(selected_samples):
        print(f"\n[{idx+1}/{NUM_SAMPLES}] Processing sample...")
        
        frame_paths = sample['frame_paths'][:4]
        gt_traj = np.array(sample['trajectory'])
        gt_reasoning = sample['reasoning']
        
        # Load last frame for visualization
        img_path = frame_paths[-1]
        img = np.array(Image.open(img_path).convert("RGB"))
        
        # Original model
        print("  Running Original model...")
        orig_traj, orig_reasoning = run_inference(
            original_model, processor_orig, frame_paths, DEVICE_ORIGINAL
        )
        
        # Fine-tuned model
        print("  Running Fine-tuned model...")
        ft_traj, ft_reasoning = run_inference(
            finetuned_model, processor_ft, frame_paths, DEVICE_FINETUNED
        )
        
        results.append({
            "img": img,
            "gt_traj": gt_traj,
            "orig_traj": orig_traj,
            "ft_traj": ft_traj,
            "gt_reasoning": gt_reasoning,
            "orig_reasoning": orig_reasoning,
            "ft_reasoning": ft_reasoning,
        })
    
    # Create grid visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)
    
    cols = 5
    rows = (NUM_SAMPLES + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(25, 5 * rows))
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        ax = axes[i]
        img = result["img"]
        gt_traj = result["gt_traj"]
        orig_traj = result["orig_traj"]
        ft_traj = result["ft_traj"]
        
        # Display image
        ax.imshow(img)
        
        # Project trajectories
        gt_pts = project_to_image(gt_traj[:, :2], img.shape)
        orig_pts = project_to_image(orig_traj, img.shape) if orig_traj is not None else []
        ft_pts = project_to_image(ft_traj, img.shape) if ft_traj is not None else []
        
        # Plot GT (Green)
        if len(gt_pts) > 1:
            xs, ys = zip(*gt_pts)
            ax.plot(xs, ys, 'g--', linewidth=4, alpha=0.8, label='GT')
        
        # Plot Original (Blue)
        if len(orig_pts) > 1:
            xs, ys = zip(*orig_pts)
            ax.plot(xs, ys, 'b-', linewidth=4, alpha=0.7, label='Original')
        
        # Plot Fine-tuned (Red)
        if len(ft_pts) > 1:
            xs, ys = zip(*ft_pts)
            ax.plot(xs, ys, 'r-', linewidth=4, alpha=0.9, label='Fine-tuned')
        
        # Title with short reasoning
        gt_short = result["gt_reasoning"][:50] + "..." if len(result["gt_reasoning"]) > 50 else result["gt_reasoning"]
        ax.set_title(f"Sample #{i+1}\n{gt_short}", fontsize=10)
        ax.axis('off')
        
        if i == 0:
            ax.legend(loc='lower right', fontsize=9)
    
    # Hide unused subplots
    for j in range(NUM_SAMPLES, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Original vs Fine-tuned Alpamayo: Trajectory Comparison', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    viz_path = os.path.join(OUTPUT_DIR, "comparison_grid.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Grid visualization saved to {viz_path}")
    
    # Save individual detailed comparisons
    print("\nCreating individual detailed comparisons...")
    for i, result in enumerate(results[:5]):  # Save first 5 in detail
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Image with trajectories
        ax_img = axes[0]
        img = result["img"]
        ax_img.imshow(img)
        
        gt_traj = result["gt_traj"]
        orig_traj = result["orig_traj"]
        ft_traj = result["ft_traj"]
        
        gt_pts = project_to_image(gt_traj[:, :2], img.shape)
        orig_pts = project_to_image(orig_traj, img.shape) if orig_traj is not None else []
        ft_pts = project_to_image(ft_traj, img.shape) if ft_traj is not None else []
        
        if len(gt_pts) > 1:
            xs, ys = zip(*gt_pts)
            ax_img.plot(xs, ys, 'g--', linewidth=5, alpha=0.8, label='Ground Truth')
        if len(orig_pts) > 1:
            xs, ys = zip(*orig_pts)
            ax_img.plot(xs, ys, 'b-', linewidth=5, alpha=0.7, label='Original Model')
        if len(ft_pts) > 1:
            xs, ys = zip(*ft_pts)
            ax_img.plot(xs, ys, 'r-', linewidth=5, alpha=0.9, label='Fine-tuned Model')
        
        ax_img.set_title(f'Sample #{i+1}: Camera View', fontsize=14, fontweight='bold')
        ax_img.axis('off')
        ax_img.legend(loc='lower right', fontsize=11)
        
        # Right: Reasoning comparison
        ax_txt = axes[1]
        ax_txt.axis('off')
        
        def truncate(text, max_len=300):
            return text[:max_len] + "..." if len(text) > max_len else text
        
        txt = f"Ground Truth Reasoning:\n{truncate(result['gt_reasoning'])}\n\n"
        txt += f"Original Model Reasoning:\n{truncate(result['orig_reasoning'])}\n\n"
        txt += f"Fine-tuned Model Reasoning:\n{truncate(result['ft_reasoning'])}"
        
        ax_txt.text(0, 0.95, txt, fontsize=9, verticalalignment='top',
                   wrap=True, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_txt.set_title('Reasoning Comparison', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        detail_path = os.path.join(OUTPUT_DIR, f"detail_sample_{i:02d}.png")
        plt.savefig(detail_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {detail_path}")
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
