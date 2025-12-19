import json
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
DATA_PATH = Path("/home/byounggun/alpamayo/src/alpamayo_r1/alignment/preference_dataset/finetune_data.jsonl")
OUTPUT_PATH = Path("/home/byounggun/alpamayo/src/alpamayo_r1/alignment/preference_dataset")
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
        # JSONL has "original_trajectory" and "trajectory" (which is the corrected one)
        orig_traj = sample["original_trajectory"]
        corr_traj = sample["trajectory"]
        
        orig_pts = project_to_image(orig_traj, img_np.shape)
        corr_pts = project_to_image(corr_traj, img_np.shape)
        
        # Plot Original (Blue)
        if len(orig_pts) > 1:
            xs, ys = zip(*orig_pts)
            ax.plot(xs, ys, 'b-', linewidth=4, alpha=0.7, label='Original (Violated)')
            
        # Plot Corrected (Red)
        if len(corr_pts) > 1:
            xs, ys = zip(*corr_pts)
            ax.plot(xs, ys, 'r--', linewidth=4, alpha=0.9, label='Corrected (Target)')
            
        # Meta info
        expl = sample.get("explanation", "No explanation")
        # Truncate explanation
        short_expl = (expl[:50] + '...') if len(expl) > 50 else expl
        ax.set_title(f"Sample #{i+1}\n{short_expl}", fontsize=10, color='red')
        ax.axis('off')
        
        if i == 0:
            ax.legend(loc='lower right')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    viz_path = OUTPUT_PATH / "training_data_viz.png"
    plt.savefig(viz_path)
    print(f"Visualization saved to {viz_path}")

if __name__ == "__main__":
    main()
