#!/usr/bin/env python3
"""
Alpamayo inference on Rellis-3D using sequential frames.
Uses 4 consecutive frames from single camera.
"""

import sys
import json
import copy
from pathlib import Path

sys.path.insert(0, '/home/byounggun/alpamayo/src')

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

print("="*60)
print("RELLIS-3D ALPAMAYO INFERENCE")
print("="*60)

# Configuration
NUM_SAMPLES = 5
NUM_FRAMES = 4  # Use 4 sequential frames like PhysicalAI-AV
IMAGE_DIR = Path("/home/byounggun/alpamayo/Rellis-3D/00000/pylon_camera_node")
OUTPUT_DIR = Path("/home/byounggun/alpamayo/src/alpamayo_r1/alignment/rellis3d_test")
OUTPUT_DIR.mkdir(exist_ok=True)

# Camera intrinsics from Rellis-3D
FX, FY = 2813.64, 2808.33
CX, CY = 969.29, 624.05

# Get sorted image list
all_images = sorted(IMAGE_DIR.glob("*.jpg"))
print(f"Found {len(all_images)} images")

# Select samples - need enough gap for 4 frames each
sample_starts = np.linspace(0, len(all_images) - NUM_FRAMES - 1, NUM_SAMPLES, dtype=int)
print(f"Testing {NUM_SAMPLES} samples (4 frames each)")

# ==============================================================================
# Load Alpamayo Model
# ==============================================================================
print("\nLoading Alpamayo-R1 model...")
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to(device)
processor = helper.get_processor(model.tokenizer)
print(f"✓ Model loaded on {device}")

# ==============================================================================
# Load GPT Critic
# ==============================================================================
print("Loading GPT critic...")
from alpamayo_r1.alignment.openrouter_critic import OpenRouterCritic
from alpamayo_r1.alignment.guidelines import load_guidelines

critic = OpenRouterCritic(model="openai/gpt-4o")  # Better image analysis
guidelines = load_guidelines()
guidelines_prompt = guidelines.to_prompt()
print(f"✓ Loaded {len(guidelines)} off-road rules")

# ==============================================================================
# Helper: rotate 90 CCW for BEV
# ==============================================================================
def rotate_90cc(xy):
    return np.stack([-xy[1], xy[0]], axis=0)

def project_to_image(trajectory_xy, img_shape):
    """Project trajectory to image pixels."""
    H, W = img_shape[:2]
    points = []
    
    for x_fwd, y_lat in trajectory_xy:
        if x_fwd <= 1.0:
            continue
        # Simple ground plane projection
        u = CX - FX * y_lat / x_fwd
        v = CY + FY * 1.5 / x_fwd  # camera height ~1.5m
        
        if 0 <= u < W and 0 <= v < H:
            points.append((int(u), int(v)))
    
    return points

# ==============================================================================
# Run Inference
# ==============================================================================
print(f"\nProcessing {NUM_SAMPLES} samples...")
results = []
all_trajectories = []
all_corrected_trajectories = []
sample_images = []

for i, start_idx in enumerate(sample_starts):
    print(f"\n[{i+1}/{NUM_SAMPLES}] Frames {start_idx} to {start_idx + NUM_FRAMES - 1}")
    
    try:
        # Load 4 sequential frames
        frame_paths = [all_images[start_idx + j] for j in range(NUM_FRAMES)]
        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            img_np = np.array(img)
            # Convert to (C, H, W) and normalize
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            frames.append(img_t)
        
        # Stack: (4, C, H, W)
        frames_tensor = torch.stack(frames, dim=0)
        print(f"  Input shape: {frames_tensor.shape}")
        
        # Create message (like notebook)
        messages = helper.create_message(frames_tensor)
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Realistic ego history: straight line motion at ~5 m/s
        # 16 history steps, 0.1s each = 1.6s history
        num_history = 16
        dt = 0.1
        speed = 5.0  # m/s
        
        # Position history: moving forward in straight line
        # At t=0, we're at origin. Going back in time...
        times = np.arange(-num_history + 1, 1) * dt  # [-1.5, -1.4, ..., -0.1, 0.0]
        positions = np.zeros((num_history, 3))
        positions[:, 0] = times * speed  # x = forward
        positions[:, 1] = 0  # y = 0 (straight line)
        positions[:, 2] = 0  # z = 0
        
        ego_history_xyz = torch.from_numpy(positions).float().unsqueeze(0).unsqueeze(0).to(device)
        # Shape: (1, 1, 16, 3)
        
        # Rotation: 3x3 rotation matrix (NOT quaternion!)
        # PhysicalAI-AV uses shape (B, S, T, 3, 3)
        # Identity rotation for straight line driving
        ego_history_rot = torch.eye(3).unsqueeze(0).repeat(num_history, 1, 1)  # (16, 3, 3)
        ego_history_rot = ego_history_rot.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 16, 3, 3)
        
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        model_inputs = helper.to_device(model_inputs, device)
        
        print(f"  Running Alpamayo inference...")
        
        # Run inference
        torch.cuda.manual_seed_all(42 + i)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=copy.deepcopy(model_inputs),
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=True,
            )
        
        # Extract trajectory: (T, 2) [x_forward, y_lateral]
        traj = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]  # (64, 2)
        reasoning = extra["cot"][0, 0, 0] if extra.get("cot") is not None else "No CoC"
        
        print(f"  Trajectory shape: {traj.shape}")
        print(f"  CoC: {reasoning[:100]}..." if len(reasoning) > 100 else f"  CoC: {reasoning}")
        
        # Store
        all_trajectories.append(traj.copy())
        sample_images.append(np.array(Image.open(frame_paths[-1])))  # Last frame
        
        # ==============================================================================
        # GPT Critique
        # ==============================================================================
        print(f"  Getting GPT critique...")
        
        # Use last frame for critique
        img_for_critique = np.array(Image.open(frame_paths[-1]))
        
        critique = critic.critique(
            images=[img_for_critique],
            reasoning=reasoning,
            trajectory=traj,
            guidelines=guidelines_prompt,
        )
        
        status = "VIOLATED" if critique.violated else "OK"
        explanation = critique.explanation or "No explanation"
        print(f"  → {status}: {explanation[:60]}...")
        
        if critique.corrected_trajectory is not None:
            all_corrected_trajectories.append(critique.corrected_trajectory)
        else:
            all_corrected_trajectories.append(traj.copy())
        
        results.append({
            "frames": [p.name for p in frame_paths],
            "alpamayo_reasoning": reasoning,
            "trajectory": traj.tolist(),
            "violated": critique.violated,
            "explanation": critique.explanation,
            "corrected_reasoning": critique.corrected_reasoning,
            "corrected_trajectory": critique.corrected_trajectory.tolist() if critique.corrected_trajectory is not None else None,
        })
        
        # Save last frame
        Image.open(frame_paths[-1]).save(OUTPUT_DIR / f"sample_{i:02d}.jpg")
        
    except Exception as e:
        import traceback
        print(f"  → ERROR: {e}")
        traceback.print_exc()
        results.append({"error": str(e)})
        all_trajectories.append(np.zeros((64, 2)))
        all_corrected_trajectories.append(np.zeros((64, 2)))
        sample_images.append(np.zeros((480, 640, 3), dtype=np.uint8))

# ==============================================================================
# Save Results
# ==============================================================================
with open(OUTPUT_DIR / "alpamayo_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ==============================================================================
# VISUALIZATION: Image + Trajectory Projection + BEV
# ==============================================================================
print("\nCreating visualization...")

fig = plt.figure(figsize=(20, 4 * NUM_SAMPLES))

for i in range(NUM_SAMPLES):
    if i >= len(results) or "error" in results[i]:
        continue
    
    # Left: Image with projected trajectory
    ax_img = fig.add_subplot(NUM_SAMPLES, 3, i*3 + 1)
    img = sample_images[i]
    ax_img.imshow(img)
    
    # Project trajectories onto image
    orig_pts = project_to_image(all_trajectories[i], img.shape)
    corr_pts = project_to_image(all_corrected_trajectories[i], img.shape)
    
    if len(orig_pts) > 1:
        xs, ys = zip(*orig_pts)
        ax_img.plot(xs, ys, 'b-', linewidth=4, alpha=0.8, label='Alpamayo')
        ax_img.scatter(xs, ys, c='cyan', s=30, edgecolors='blue', zorder=5)
    
    if len(corr_pts) > 1:
        xs, ys = zip(*corr_pts)
        ax_img.plot(xs, ys, 'r--', linewidth=3, alpha=0.8, label='Corrected')
        ax_img.scatter(xs, ys, c='orange', s=30, edgecolors='red', zorder=5, marker='x')
    
    status = "VIOLATED" if results[i].get("violated") else "OK"
    status_color = 'red' if results[i].get("violated") else 'green'
    ax_img.set_title(f"#{i+1} {status}", color=status_color, fontweight='bold', fontsize=12)
    ax_img.axis('off')
    if i == 0:
        ax_img.legend(loc='upper right', fontsize=9)
    
    # Center: BEV plot
    ax_bev = fig.add_subplot(NUM_SAMPLES, 3, i*3 + 2)
    traj = all_trajectories[i]
    corr = all_corrected_trajectories[i]
    
    pred_rot = rotate_90cc(traj.T)
    corr_rot = rotate_90cc(corr.T)
    
    ax_bev.plot(*pred_rot, 'b-o', linewidth=2, markersize=3, label='Alpamayo')
    ax_bev.plot(*corr_rot, 'r--x', linewidth=2, markersize=3, label='Corrected')
    ax_bev.scatter([0], [0], c='green', s=100, marker='^', zorder=5, label='Ego')
    ax_bev.set_xlabel('x (m)')
    ax_bev.set_ylabel('y (m)')
    ax_bev.axis('equal')
    ax_bev.grid(True, alpha=0.3)
    ax_bev.set_title('BEV View')
    if i == 0:
        ax_bev.legend(loc='upper left', fontsize=8)
    
    # Right: Reasoning
    ax_txt = fig.add_subplot(NUM_SAMPLES, 3, i*3 + 3)
    ax_txt.axis('off')
    coc = (results[i].get("alpamayo_reasoning") or "N/A")[:300]
    explanation = (results[i].get("explanation") or "N/A")[:200]
    txt = f"Alpamayo CoC:\n{coc}\n\nGPT: {explanation}"
    ax_txt.text(0, 0.9, txt, fontsize=8, verticalalignment='top', 
                wrap=True, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_txt.set_title('Reasoning', fontsize=10)

plt.suptitle('Rellis-3D Alpamayo Inference + GPT Critique', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "alpamayo_on_rellis3d.png", dpi=150, bbox_inches='tight')
plt.close()

# ==============================================================================
# Summary
# ==============================================================================
violated_count = sum(1 for r in results if r.get("violated", False))
error_count = sum(1 for r in results if "error" in r)
ok_count = len(results) - violated_count - error_count

print(f"\n" + "="*60)
print("COMPLETE!")
print("="*60)
print(f"Total: {len(results)}")
print(f"  VIOLATED: {violated_count}")
print(f"  OK: {ok_count}")
print(f"  ERROR: {error_count}")
print(f"\nSaved: {OUTPUT_DIR}/alpamayo_on_rellis3d.png")
