import json
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
DATA_PATH = Path("/home/byounggun/alpamayo/src/alpamayo_r1/alignment/preference_dataset_v2/finetune_data.jsonl")
OUTPUT_PATH = Path("/home/byounggun/alpamayo/src/alpamayo_r1/alignment/preference_dataset_v2")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
NUM_SAMPLES = 20

# Camera Intrinsics
FX, FY = 2813.64, 2808.33
CX, CY = 969.29, 624.05

def project_to_image(trajectory_xy, img_shape):
    """Project trajectory to image coordinates"""
    H, W = img_shape[:2]
    points = []
    for pt in trajectory_xy:
        x_fwd, y_lat = pt[0], pt[1]
        if x_fwd <= 1.0:  # Same as create_viz.py
            continue
        u = CX - FX * y_lat / x_fwd
        v = CY + FY * 1.5 / x_fwd  # 1.5m camera height
        if 0 <= u < W and 0 <= v < H:
            points.append((int(u), int(v)))
    return points

def main():
    print(f"Loading dataset from {DATA_PATH}...")
    
    violated_samples = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            sample = json.loads(line)
            if sample.get("source") == "corrected":
                violated_samples.append(sample)
    
    print(f"Found {len(violated_samples)} corrected (violated) samples.")
    
    if len(violated_samples) < NUM_SAMPLES:
        selected_samples = violated_samples
    else:
        # Select 10 evenly spaced samples
        # indices = np.linspace(0, len(violated_samples)-1, NUM_SAMPLES, dtype=int)
        # selected_samples = [violated_samples[i] for i in indices]
        # Or just first 10? User said "pick 10"
        selected_samples = violated_samples[:NUM_SAMPLES]
        
    print(f"Visualizing {len(selected_samples)} samples...")
    
    # Create grid plot
    cols = 5
    rows = (len(selected_samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(25, 5 * rows))
    axes = axes.flatten()
    
    for i, sample in enumerate(selected_samples):
        ax = axes[i]
        
        # Load image
        # frame_paths has absolute paths
        img_path = sample["frame_paths"][-1]
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            ax.imshow(img_np)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            ax.text(0.5, 0.5, "Image Load Error", ha='center')
            continue
            
        # Project trajectories
        # V2 format: "alpamayo_trajectory" (original) and "trajectory" (final: alpamayo or corrected)
        alpamayo_traj = sample.get("alpamayo_trajectory", [])
        final_traj = sample["trajectory"]
        source = sample.get("source", "unknown")

        alpamayo_pts = project_to_image(alpamayo_traj, img_np.shape)
        final_pts = project_to_image(final_traj, img_np.shape)

        # Plot Alpamayo Original (Blue)
        if len(alpamayo_pts) > 1:
            xs, ys = zip(*alpamayo_pts)
            ax.plot(xs, ys, 'b-', linewidth=3, alpha=0.6, label='Alpamayo Original')

        # Plot Final (Red if corrected, Green if kept original)
        if len(final_pts) > 1:
            xs, ys = zip(*final_pts)
            if source == "corrected":
                ax.plot(xs, ys, 'r--', linewidth=4, alpha=0.9, label='Corrected by Critic')
            else:
                ax.plot(xs, ys, 'g-', linewidth=4, alpha=0.9, label='Kept Original')

        # Meta info
        expl = sample.get("explanation", "No explanation")
        reasoning = sample.get("reasoning", "")[:35]
        # Truncate explanation
        short_expl = (expl[:45] + '...') if len(expl) > 45 else expl
        title_color = 'red' if source == "corrected" else 'green'
        ax.set_title(f"#{i+1} [{source}]\n{reasoning}\n{short_expl}", fontsize=9, color=title_color)
        ax.axis('off')

        if i == 0:
            ax.legend(loc='lower right', fontsize=8)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    viz_path = OUTPUT_PATH / "training_data_viz.png"
    plt.savefig(viz_path)
    print(f"Visualization saved to {viz_path}")

if __name__ == "__main__":
    main()
