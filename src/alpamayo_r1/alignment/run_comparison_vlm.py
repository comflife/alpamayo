#!/usr/bin/env python3
"""
VLM-only Fine-tuning Comparison

Compares Original vs VLM-only Fine-tuned Alpamayo models.
Shows differences in:
- Text/reasoning generation
- Trajectories (even though VLM wasn't trained on trajectories)

This helps understand how VLM fine-tuning affects both language and trajectory generation.
"""

import os
import sys
import copy
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from peft import PeftModel

sys.path.insert(0, '/home/byounggun/alpamayo/src')

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# Configuration
# Point this to a PEFT adapter directory (e.g. outputs/alpamayo_vlm_v2/checkpoint-200)
CHECKPOINT_PATH = "/home/byounggun/alpamayo/outputs/alpamayo_vlm_v2/checkpoint-200"
DATA_PATH = "/home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset_v2/testing.jsonl"
OUTPUT_DIR = "/home/byounggun/alpamayo/outputs/comparison_vlm_only"
DEVICE_ORIGINAL = "cuda:0"
DEVICE_FINETUNED = "cuda:1"
NUM_SAMPLES = 20  # Number of samples to visualize

# Rellis-3D camera info directory
RELLIS_DIR = "/home/byounggun/alpamayo/Rellis-3D"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_camera_params(folder):
    """Load camera intrinsics for a specific Rellis-3D sequence."""
    camera_info_path = os.path.join(RELLIS_DIR, f"camera_info_{folder}.txt")
    if os.path.exists(camera_info_path):
        with open(camera_info_path, 'r') as f:
            fx, fy, cx, cy = map(float, f.read().strip().split())
        return fx, fy, cx, cy
    else:
        # Default values if file not found
        return 2813.64, 2808.33, 969.29, 624.05


def project_to_image(trajectory_xy, img_shape, fx, fy, cx, cy):
    """Project BEV trajectory to image coordinates."""
    H, W = img_shape[:2]
    points = []
    for x_fwd, y_lat in trajectory_xy:
        if x_fwd <= 1.0:
            continue
        u = cx - fx * y_lat / x_fwd
        v = cy + fy * 1.5 / x_fwd  # 1.5m camera height
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


def load_model(checkpoint_path=None, device="cuda:0"):
    """Load Alpamayo model (original or fine-tuned)."""
    if checkpoint_path is None:
        print(f"\nLoading ORIGINAL model on {device}...")
        model = AlpamayoR1.from_pretrained(
            "nvidia/Alpamayo-R1-10B",
            dtype=torch.bfloat16,
        ).to(device)
        print("✓ Original model loaded")
    else:
        print(f"\nLoading VLM FINE-TUNED model on {device}...")
        model = AlpamayoR1.from_pretrained(
            "nvidia/Alpamayo-R1-10B",
            dtype=torch.bfloat16,
        ).to(device)

        # Load VLM LoRA / PEFT adapter
        # Accept either:
        # - checkpoint_path pointing directly to an adapter dir (adapter_config.json present)
        # - legacy layout where adapter is in checkpoint_path/final
        candidate_paths = [
            checkpoint_path,
            os.path.join(checkpoint_path, "final"),
        ]

        vlm_lora_path = None
        for p in candidate_paths:
            if p and os.path.isdir(p) and os.path.exists(os.path.join(p, "adapter_config.json")):
                vlm_lora_path = p
                break

        if vlm_lora_path is not None:
            model.vlm = PeftModel.from_pretrained(model.vlm, vlm_lora_path)
            print(f"  ✓ VLM LoRA loaded from {vlm_lora_path}")
        else:
            print(
                "  ✗ VLM LoRA not found. Looked for adapter_config.json in: "
                + ", ".join([p for p in candidate_paths if p])
            )

        print("✓ VLM fine-tuned model loaded")

    model.eval()
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


def truncate_text(text, max_len=120):
    """Keep overlay text short for visualization."""
    return text[:max_len] + "..." if len(text) > max_len else text


def main():
    print("=" * 80)
    print("ALPAMAYO VLM FINE-TUNING COMPARISON")
    print("Original vs VLM-only Fine-tuned")
    print("=" * 80)

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
    print("\n" + "=" * 80)
    print("LOADING MODELS")
    print("=" * 80)

    original_model = load_model(checkpoint_path=None, device=DEVICE_ORIGINAL)
    processor_orig = helper.get_processor(original_model.tokenizer)

    finetuned_model = load_model(checkpoint_path=CHECKPOINT_PATH, device=DEVICE_FINETUNED)
    processor_ft = helper.get_processor(finetuned_model.tokenizer)

    # Run inference on all samples
    print("\n" + "=" * 80)
    print("RUNNING INFERENCE")
    print("=" * 80)

    results = []

    for idx, sample in enumerate(selected_samples):
        print(f"\n[{idx+1}/{NUM_SAMPLES}] Processing sample...")

        frame_paths = sample['frame_paths'][:4]
        gt_reasoning = sample['reasoning']  # Gemini-generated ground truth
        folder = sample.get('folder', 'unknown')

        # Load camera parameters for this sequence
        fx, fy, cx, cy = load_camera_params(folder)

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
            "orig_traj": orig_traj,
            "ft_traj": ft_traj,
            "gt_reasoning": gt_reasoning,
            "orig_reasoning": orig_reasoning,
            "ft_reasoning": ft_reasoning,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        })

    # Create grid visualization
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)

    cols = 5
    rows = (NUM_SAMPLES + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(25, 5 * rows))
    axes = axes.flatten()

    for i, result in enumerate(results):
        ax = axes[i]
        img = result["img"]
        orig_traj = result["orig_traj"]
        ft_traj = result["ft_traj"]
        fx, fy, cx, cy = result["fx"], result["fy"], result["cx"], result["cy"]

        # Display image
        ax.imshow(img)

        # Project trajectories
        orig_pts = project_to_image(orig_traj, img.shape, fx, fy, cx, cy) if orig_traj is not None else []
        ft_pts = project_to_image(ft_traj, img.shape, fx, fy, cx, cy) if ft_traj is not None else []

        # Plot Original (Blue)
        if len(orig_pts) > 1:
            xs, ys = zip(*orig_pts)
            ax.plot(xs, ys, 'b-', linewidth=4, alpha=0.7, label='Original')

        # Plot Fine-tuned (Red)
        if len(ft_pts) > 1:
            xs, ys = zip(*ft_pts)
            ax.plot(xs, ys, 'r-', linewidth=4, alpha=0.9, label='VLM Fine-tuned')

        # Title with short reasoning
        gt_short = result["gt_reasoning"][:50] + "..." if len(result["gt_reasoning"]) > 50 else result["gt_reasoning"]
        ax.set_title(f"Sample #{i+1}\nGT: {gt_short}", fontsize=9)
        ax.axis('off')

        # Overlay fine-tuned model text output on the image
        ft_short = truncate_text(result["ft_reasoning"], 100)
        ax.text(
            0.02, 0.02,
            f"FT: {ft_short}",
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )

        if i == 0:
            ax.legend(loc='lower right', fontsize=9)

    # Hide unused subplots
    for j in range(NUM_SAMPLES, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Original vs VLM Fine-tuned Alpamayo: Text & Trajectory Comparison',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()

    viz_path = os.path.join(OUTPUT_DIR, "comparison_grid.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Grid visualization saved to {viz_path}")

    # Save individual detailed comparisons
    print("\nCreating individual detailed comparisons...")
    for i, result in enumerate(results[:5]):  # Save first 5 in detail
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Left: Image with trajectories
        ax_img = axes[0]
        img = result["img"]
        ax_img.imshow(img)

        orig_traj = result["orig_traj"]
        ft_traj = result["ft_traj"]
        fx, fy, cx, cy = result["fx"], result["fy"], result["cx"], result["cy"]

        orig_pts = project_to_image(orig_traj, img.shape, fx, fy, cx, cy) if orig_traj is not None else []
        ft_pts = project_to_image(ft_traj, img.shape, fx, fy, cx, cy) if ft_traj is not None else []

        if len(orig_pts) > 1:
            xs, ys = zip(*orig_pts)
            ax_img.plot(xs, ys, 'b-', linewidth=5, alpha=0.7, label='Original Model')
        if len(ft_pts) > 1:
            xs, ys = zip(*ft_pts)
            ax_img.plot(xs, ys, 'r-', linewidth=5, alpha=0.9, label='VLM Fine-tuned Model')

        ft_short = truncate_text(result["ft_reasoning"], 140)
        ax_img.text(
            0.02, 0.02,
            f"FT: {ft_short}",
            transform=ax_img.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax_img.set_title(f'Sample #{i+1}: Camera View', fontsize=14, fontweight='bold')
        ax_img.axis('off')
        ax_img.legend(loc='lower right', fontsize=11)

        # Right: Reasoning comparison
        ax_txt = axes[1]
        ax_txt.axis('off')

        def truncate(text, max_len=400):
            return text[:max_len] + "..." if len(text) > max_len else text

        txt = "═" * 60 + "\n"
        txt += "GROUND TRUTH (Gemini-generated):\n"
        txt += "─" * 60 + "\n"
        txt += f"{truncate(result['gt_reasoning'], 350)}\n\n"

        txt += "═" * 60 + "\n"
        txt += "ORIGINAL MODEL:\n"
        txt += "─" * 60 + "\n"
        txt += f"{truncate(result['orig_reasoning'], 350)}\n\n"

        txt += "═" * 60 + "\n"
        txt += "VLM FINE-TUNED MODEL:\n"
        txt += "─" * 60 + "\n"
        txt += f"{truncate(result['ft_reasoning'], 350)}"

        ax_txt.text(0.05, 0.95, txt, fontsize=8, verticalalignment='top',
                   wrap=True, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax_txt.set_title('Reasoning Comparison', fontsize=14, fontweight='bold')

        plt.tight_layout()
        detail_path = os.path.join(OUTPUT_DIR, f"detail_sample_{i:02d}.png")
        plt.savefig(detail_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {detail_path}")

    # Create text comparison summary
    print("\nCreating text comparison summary...")
    summary_path = os.path.join(OUTPUT_DIR, "text_comparison.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ALPAMAYO VLM FINE-TUNING: TEXT COMPARISON\n")
        f.write("=" * 80 + "\n\n")

        for i, result in enumerate(results):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"SAMPLE #{i+1}\n")
            f.write(f"{'=' * 80}\n\n")

            f.write("GROUND TRUTH (Gemini):\n")
            f.write(f"{result['gt_reasoning']}\n\n")

            f.write("ORIGINAL MODEL:\n")
            f.write(f"{result['orig_reasoning']}\n\n")

            f.write("VLM FINE-TUNED MODEL:\n")
            f.write(f"{result['ft_reasoning']}\n\n")

    print(f"✓ Text comparison saved to {summary_path}")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  - comparison_grid.png (20 samples overview)")
    print(f"  - detail_sample_00.png to detail_sample_04.png (detailed view)")
    print(f"  - text_comparison.txt (full text comparison)")


if __name__ == "__main__":
    main()
