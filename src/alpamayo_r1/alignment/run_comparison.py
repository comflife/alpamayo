#!/usr/bin/env python3
"""
Run inference comparison between Original and Fine-tuned models and visualize results.
"""

import os
import sys
import copy
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from peft import PeftModel, LoraConfig, get_peft_model

sys.path.insert(0, '/home/byounggun/alpamayo/src')

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# Paths
CHECKPOINT_PATH = "/home/byounggun/alpamayo/outputs/alpamayo_full_finetuned"
DATA_PATH = "/home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset/finetune_data.jsonl"
OUTPUT_DIR = "/home/byounggun/alpamayo/outputs/comparison_visualizations"
DEVICE_ORIGINAL = "cuda:0"  # Original model on GPU 0
DEVICE_FINETUNED = "cuda:1"  # Fine-tuned model on GPU 1

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def load_original_model():
    """Load original model in bfloat16."""
    print("\n" + "=" * 60)
    print(f"Loading ORIGINAL Alpamayo model on {DEVICE_ORIGINAL}...")
    print("=" * 60)

    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
    )
    model = model.to(DEVICE_ORIGINAL)
    model.eval()

    print("Original model loaded!")
    return model


def load_finetuned_model():
    """Load fine-tuned model with VLM LoRA + Expert LoRA + Diffusion decoder weights."""
    print("\n" + "=" * 60)
    print(f"Loading FINE-TUNED Alpamayo model on {DEVICE_FINETUNED}...")
    print("=" * 60)

    # Load base model
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
    )
    model = model.to(DEVICE_FINETUNED)

    # Load VLM LoRA
    vlm_lora_path = os.path.join(CHECKPOINT_PATH, "final", "vlm_lora")
    if os.path.exists(vlm_lora_path):
        print(f"Loading VLM LoRA from {vlm_lora_path}...")
        model.vlm = PeftModel.from_pretrained(model.vlm, vlm_lora_path)
        print("  ✓ VLM LoRA loaded!")
    else:
        print(f"  ⚠ Warning: VLM LoRA not found, using base VLM")

    # Load Expert LoRA + Diffusion decoder weights
    checkpoint_path = os.path.join(CHECKPOINT_PATH, "final", "expert_diffusion.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading Expert + Diffusion weights...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE_FINETUNED)

        # Apply LoRA to Expert
        if "expert_lora" in checkpoint:
            expert_lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            model.expert = get_peft_model(model.expert, expert_lora_config)
            model.expert.load_state_dict(checkpoint["expert_lora"], strict=False)
            print("  ✓ Expert LoRA loaded!")

        if "action_in_proj" in checkpoint:
            model.action_in_proj.load_state_dict(checkpoint["action_in_proj"])
            print("  ✓ action_in_proj loaded!")

        if "action_out_proj" in checkpoint:
            model.action_out_proj.load_state_dict(checkpoint["action_out_proj"])
            print("  ✓ action_out_proj loaded!")
    else:
        print(f"  ⚠ Warning: checkpoint not found")

    model.eval()
    print("Fine-tuned model loaded successfully!")
    return model


def run_inference(model, processor, frame_paths, device):
    """Run inference and return predicted trajectory."""
    # Load frames
    frames = []
    for fp in frame_paths:
        if os.path.exists(fp):
            img = Image.open(fp).convert("RGB")
            img_np = np.array(img)
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            frames.append(img_t)

    if not frames:
        print("ERROR: No images loaded!")
        return None

    # Create message and tokenize
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

    # Create ego history
    ego_history_xyz, ego_history_rot = create_ego_history(device)

    # Prepare model inputs
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    model_inputs = helper.to_device(model_inputs, device)

    # Run inference
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

    return pred_xyz, extra


def rotate_90cc(xy):
    """Rotate 90 degrees counter-clockwise for BEV."""
    return np.stack([-xy[1], xy[0]], axis=0)


def visualize_comparison(sample_idx, frame_paths, gt_traj, gt_reasoning,
                        original_pred, original_reasoning,
                        finetuned_pred, finetuned_reasoning, save_path):
    """Create comparison visualization in 3-column format: Image | BEV | Reasoning."""
    fig = plt.figure(figsize=(20, 6))

    # Load last frame (like test_rellis3d.py)
    img = None
    if os.path.exists(frame_paths[-1]):
        img = np.array(Image.open(frame_paths[-1]))

    # Left: Image
    ax_img = fig.add_subplot(1, 3, 1)
    if img is not None:
        ax_img.imshow(img)
    ax_img.set_title(f"Sample {sample_idx}: Input Frame", fontsize=14, fontweight='bold')
    ax_img.axis('off')

    # Center: BEV plot
    ax_bev = fig.add_subplot(1, 3, 2)
    gt_traj_np = np.array(gt_traj)

    # Rotate trajectories 90 degrees CCW for BEV
    gt_rot = rotate_90cc(gt_traj_np.T)
    ax_bev.plot(*gt_rot, 'g--o', linewidth=2, markersize=3, label='Ground Truth', alpha=0.7)

    if original_pred is not None:
        # Extract trajectory: shape is (B, S, num_samples, T, 3)
        # Use [0, 0, 0, :, :2] like test_rellis3d.py
        orig_traj = original_pred[0, 0, 0, :, :2].cpu().numpy()
        orig_rot = rotate_90cc(orig_traj.T)
        ax_bev.plot(*orig_rot, 'b-o', linewidth=2, markersize=3, label='Original', alpha=0.8)

    if finetuned_pred is not None:
        ft_traj = finetuned_pred[0, 0, 0, :, :2].cpu().numpy()
        ft_rot = rotate_90cc(ft_traj.T)
        ax_bev.plot(*ft_rot, 'r--x', linewidth=2, markersize=3, label='Fine-tuned', alpha=0.8)

    # Ego vehicle at origin
    ax_bev.scatter([0], [0], c='cyan', s=100, marker='^', zorder=5, label='Ego', edgecolors='blue')

    ax_bev.set_xlabel('x (m)', fontsize=12)
    ax_bev.set_ylabel('y (m)', fontsize=12)
    ax_bev.set_title('BEV View', fontsize=14, fontweight='bold')
    ax_bev.grid(True, alpha=0.3)
    ax_bev.axis('equal')
    ax_bev.legend(loc='upper left', fontsize=10)

    # Right: Reasoning text
    ax_txt = fig.add_subplot(1, 3, 3)
    ax_txt.axis('off')

    # Truncate reasoning if too long
    def truncate(text, max_len=250):
        return text[:max_len] + "..." if len(text) > max_len else text

    gt_text = truncate(gt_reasoning)
    orig_text = truncate(original_reasoning) if original_reasoning else "N/A"
    ft_text = truncate(finetuned_reasoning) if finetuned_reasoning else "N/A"

    txt = f"Ground Truth:\n{gt_text}\n\n"
    txt += f"Original Model:\n{orig_text}\n\n"
    txt += f"Fine-tuned Model:\n{ft_text}"

    ax_txt.text(0, 0.98, txt, fontsize=8, verticalalignment='top',
                wrap=True, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_txt.set_title('Reasoning Comparison', fontsize=12, fontweight='bold')

    plt.suptitle(f'Sample {sample_idx}: Original vs Fine-tuned Comparison',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to {save_path}")


def main():
    print("="*60)
    print("ALPAMAYO MODEL COMPARISON")
    print("="*60)

    # Load test samples
    print("\nLoading test samples...")
    with open(DATA_PATH, "r") as f:
        samples = [json.loads(line) for line in f]

    # Select diverse samples (more spread out)
    num_samples = 10
    total_samples = len(samples)
    indices = np.linspace(0, total_samples - 1, num_samples, dtype=int)
    test_samples = [samples[i] for i in indices]
    print(f"Selected {num_samples} samples from {total_samples} total samples")

    # Load models
    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)

    original_model = load_original_model()
    processor_orig = helper.get_processor(original_model.tokenizer)

    finetuned_model = load_finetuned_model()
    processor_ft = helper.get_processor(finetuned_model.tokenizer)

    # Run inference on each sample
    print("\n" + "="*60)
    print("RUNNING INFERENCE")
    print("="*60)

    for idx, sample in enumerate(test_samples):
        print(f"\n--- Sample {idx} ---")

        frame_paths = sample['frame_paths'][:4]
        gt_traj = sample['trajectory']
        gt_reasoning = sample['reasoning']

        # Original model
        print("Running Original model...")
        original_pred, original_extra = run_inference(original_model, processor_orig, frame_paths, device=DEVICE_ORIGINAL)
        original_reasoning = original_extra["cot"][0, 0, 0] if original_extra.get("cot") is not None else "No reasoning"

        # Fine-tuned model
        print("Running Fine-tuned model...")
        finetuned_pred, finetuned_extra = run_inference(finetuned_model, processor_ft, frame_paths, device=DEVICE_FINETUNED)
        finetuned_reasoning = finetuned_extra["cot"][0, 0, 0] if finetuned_extra.get("cot") is not None else "No reasoning"

        # Visualize
        save_path = os.path.join(OUTPUT_DIR, f"comparison_sample_{idx}.png")
        visualize_comparison(idx, frame_paths, gt_traj, gt_reasoning,
                           original_pred, original_reasoning,
                           finetuned_pred, finetuned_reasoning, save_path)

    print("\n" + "="*60)
    print("COMPARISON COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
