#!/usr/bin/env python3
"""
Simple visualization script to compare Original vs Fine-tuned Alpamayo trajectories.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

sys.path.insert(0, '/home/byounggun/alpamayo/src')

# Paths
DATA_PATH = "/home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset/finetune_data.jsonl"
OUTPUT_DIR = "/home/byounggun/alpamayo/outputs/comparison_visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_sample(data_path, idx=0):
    """Load a test sample from the dataset."""
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    return None


def visualize_trajectory_comparison(sample, idx, save_path):
    """
    Visualize ground truth trajectory and reasoning on the image.
    Since we don't have model predictions yet, we'll just show the GT.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Load first frame
    frame_path = sample['frame_paths'][0]
    if os.path.exists(frame_path):
        img = Image.open(frame_path)

        # Show image on left
        axes[0].imshow(img)
        axes[0].set_title(f"Sample {idx}: First Frame", fontsize=14, fontweight='bold')
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, f"Image not found:\n{frame_path}",
                    ha='center', va='center', fontsize=10)
        axes[0].axis('off')

    # Plot trajectory on right
    gt_traj = np.array(sample['trajectory'])

    # BEV trajectory plot
    ax = axes[1]
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-', linewidth=2, label='Ground Truth', marker='o', markersize=3)

    # Mark start and end
    ax.plot(gt_traj[0, 0], gt_traj[0, 1], 'go', markersize=10, label='Start')
    ax.plot(gt_traj[-1, 0], gt_traj[-1, 1], 'rs', markersize=10, label='End')

    # Add ego vehicle at origin
    ego_width = 2.0
    ego_length = 4.0
    ego_rect = patches.Rectangle(
        (-ego_width/2, -ego_length/2),
        ego_width, ego_length,
        linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.5
    )
    ax.add_patch(ego_rect)
    ax.text(0, 0, 'EGO', ha='center', va='center', fontsize=12, fontweight='bold')

    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Bird\'s Eye View Trajectory', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='best')

    # Add reasoning text below
    reasoning = sample['reasoning']
    fig.text(0.5, 0.02, f"Reasoning: {reasoning}",
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to {save_path}")


def create_multi_sample_comparison(data_path, num_samples=5):
    """Create visualizations for multiple samples."""
    print(f"Creating visualizations for {num_samples} samples...")

    with open(data_path, "r") as f:
        samples = [json.loads(line) for line in f]

    # Select samples to visualize (first few)
    selected_samples = samples[:num_samples]

    for idx, sample in enumerate(selected_samples):
        save_path = os.path.join(OUTPUT_DIR, f"sample_{idx}_comparison.png")
        visualize_trajectory_comparison(sample, idx, save_path)

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")

    # Create summary statistics
    create_dataset_summary(samples)


def create_dataset_summary(samples):
    """Create summary statistics visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Trajectory length distribution
    traj_lengths = [len(s['trajectory']) for s in samples]
    axes[0, 0].hist(traj_lengths, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Trajectory Length (waypoints)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Trajectory Length Distribution')
    axes[0, 0].axvline(np.mean(traj_lengths), color='r', linestyle='--',
                       label=f'Mean: {np.mean(traj_lengths):.1f}')
    axes[0, 0].legend()

    # 2. Max X distance distribution
    max_x_dists = [max(abs(pt[0]) for pt in s['trajectory']) for s in samples]
    axes[0, 1].hist(max_x_dists, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Max Forward Distance (meters)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Forward Distance Distribution')
    axes[0, 1].axvline(np.mean(max_x_dists), color='r', linestyle='--',
                       label=f'Mean: {np.mean(max_x_dists):.1f}m')
    axes[0, 1].legend()

    # 3. Lateral deviation (max abs Y)
    max_y_dists = [max(abs(pt[1]) for pt in s['trajectory']) for s in samples]
    axes[1, 0].hist(max_y_dists, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Max Lateral Deviation (meters)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Lateral Deviation Distribution')
    axes[1, 0].axvline(np.mean(max_y_dists), color='r', linestyle='--',
                       label=f'Mean: {np.mean(max_y_dists):.1f}m')
    axes[1, 0].legend()

    # 4. Reasoning length distribution
    reasoning_lengths = [len(s['reasoning'].split()) for s in samples]
    axes[1, 1].hist(reasoning_lengths, bins=30, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Reasoning Length (words)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reasoning Text Length Distribution')
    axes[1, 1].axvline(np.mean(reasoning_lengths), color='r', linestyle='--',
                       label=f'Mean: {np.mean(reasoning_lengths):.1f} words')
    axes[1, 1].legend()

    plt.suptitle(f'Dataset Summary Statistics (N={len(samples)})',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    summary_path = os.path.join(OUTPUT_DIR, "dataset_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved dataset summary to {summary_path}")

    # Print text summary
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total samples: {len(samples)}")
    print(f"Trajectory length: {np.mean(traj_lengths):.1f} ± {np.std(traj_lengths):.1f} waypoints")
    print(f"Forward distance: {np.mean(max_x_dists):.1f} ± {np.std(max_x_dists):.1f} meters")
    print(f"Lateral deviation: {np.mean(max_y_dists):.1f} ± {np.std(max_y_dists):.1f} meters")
    print(f"Reasoning length: {np.mean(reasoning_lengths):.1f} ± {np.std(reasoning_lengths):.1f} words")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("ALPAMAYO DATASET VISUALIZATION")
    print("="*60)

    create_multi_sample_comparison(DATA_PATH, num_samples=10)

    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"Check output directory: {OUTPUT_DIR}")
    print("="*60)
