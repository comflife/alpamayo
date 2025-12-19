import json
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Load data
DATA_PATH = Path("/home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset/finetune_data.jsonl")
OUTPUT_PATH = Path("/home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset")

# Camera intrinsics
FX, FY = 2813.64, 2808.33
CX, CY = 969.29, 624.05

def project_to_image(trajectory_xy, img_shape):
    H, W = img_shape[:2]
    points = []
    for x_fwd, y_lat in trajectory_xy:
        if x_fwd <= 1.0:
            continue
        u = CX - FX * y_lat / x_fwd
        v = CY + FY * 1.5 / x_fwd
        if 0 <= u < W and 0 <= v < H:
            points.append((int(u), int(v)))
    return points

# Load samples
with open(DATA_PATH) as f:
    samples = [json.loads(line) for line in f]

print(f"Loaded {len(samples)} samples")

# Create visualization
n_samples = len(samples)
fig, axes = plt.subplots(5, 4, figsize=(24, 25))
axes = axes.flatten()

for i, sample in enumerate(samples):
    if i >= 20: break
    
    ax = axes[i]
    
    # Load last frame
    frame_path = sample["frame_paths"][-1]
    img = np.array(Image.open(frame_path))
    ax.imshow(img)
    
    # Project trajectories
    orig_traj = np.array(sample["original_trajectory"])
    final_traj = np.array(sample["trajectory"])
    
    orig_pts = project_to_image(orig_traj, img.shape)
    final_pts = project_to_image(final_traj, img.shape)
    
    # Draw original (Blue)
    if len(orig_pts) > 1:
        xs, ys = zip(*orig_pts)
        ax.plot(xs, ys, 'b-', linewidth=4, alpha=0.6, label='Original')
    
    # Draw corrected (Red) - only if source is "corrected"
    if sample["source"] == "corrected" and len(final_pts) > 1:
        xs, ys = zip(*final_pts)
        ax.plot(xs, ys, 'r--', linewidth=4, alpha=0.9, label='Corrected')
    elif sample["source"] == "alpamayo" and len(final_pts) > 1:
        # If OK, draw final as green to show it's good
        xs, ys = zip(*final_pts)
        ax.plot(xs, ys, 'g-', linewidth=4, alpha=0.9, label='OK')

    # Title
    is_violated = sample["source"] == "corrected"
    status = "★ CORRECTED" if is_violated else "✓ OK"
    color = 'red' if is_violated else 'green'
    
    # Shorten explanation for title
    expl = sample.get("explanation", "")
    short_expl = (expl[:40] + '...') if len(expl) > 40 else expl
    
    ax.set_title(f"#{i+1} {status}\n{short_expl}", fontsize=10, color=color, fontweight='bold')
    ax.axis('off')
    
    if i == 0:
        ax.legend(loc='upper right', fontsize=12)

plt.suptitle(f'Fine-Tuning Data: 17 Corrected / 3 OK', fontsize=20, y=1.01)
plt.tight_layout()
viz_path = OUTPUT_PATH / "dataset_visualization.png"
plt.savefig(viz_path, dpi=120, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved visualization to {viz_path}")
