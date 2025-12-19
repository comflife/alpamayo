import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from peft import PeftModel

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# Setup
FOLDER_PATH = Path("/home/byounggun/alpamayo/Rellis-3D/00003/pylon_camera_node")
ADAPTER_PATH = "/home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_checkpoints" # Uses adapter_model.safetensors
NUM_SAMPLES = 10
NUM_FRAMES = 4
OUTPUT_DIR = Path("/home/byounggun/alpamayo/src/alpamayo_r1/alignment/verification_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Camera Params for Viz
FX, FY = 2813.64, 2808.33
CX, CY = 969.29, 624.05

def project_traj(trajectory, img_shape):
    H, W = img_shape[:2]
    points = []
    # trajectory shape: (1, 1, 16, 3) -> we need (16, 3)
    # Actually Alpamayo output is (B, ns, nj, T, 3) -> (1, 1, 1, 16, 3)
    traj_np = trajectory.squeeze().cpu().numpy()
    
    if traj_np.ndim > 2:
         # take first sample if multiple dimensions remain
         traj_np = traj_np[0] 
         
    for pt in traj_np:
        x_fwd, y_lat = pt[0], pt[1]
        if x_fwd <= 1.0: continue
        u = CX - FX * y_lat / x_fwd
        v = CY + FY * 1.5 / x_fwd # 1.5m camera height approx
        if 0 <= u < W and 0 <= v < H:
            points.append((int(u), int(v)))
    return points

def load_models():
    print("Loading Original Model on CUDA:0...")
    model_orig = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B", 
        dtype=torch.bfloat16
    ).to("cuda:0")
    model_orig.eval()
    
    print("Loading Fine-Tuned Model on CUDA:1...")
    model_tuned = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B", 
        dtype=torch.bfloat16
    ).to("cuda:1")
    
    # Load Adapter
    print(f"Loading Adapter from {ADAPTER_PATH}...")
    model_tuned.vlm = PeftModel.from_pretrained(model_tuned.vlm, ADAPTER_PATH)
    model_tuned.eval()
    
    return model_orig, model_tuned

def prepare_input(image_paths, device):
    frames = []
    for fp in image_paths:
        img = Image.open(fp).convert("RGB")
        img_np = np.array(img)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        frames.append(img_t)
    
    frames_tensor = torch.stack(frames, dim=0) # (4, C, H, W)
    messages = helper.create_message(frames_tensor)
    
    # Processor part
    # We need processor from the model. 
    # Helper uses a global processor or we need to instantiate one.
    # helper.get_processor()
    # BUT wait, helper might not have get_processor exposed directly if I didn't verify helper.py
    # In test_rellis3d.py: 
    # processor = helper.get_processor(model.tokenizer)
    
    return frames_tensor, messages

def get_history_tensors(num_history=16, dt=0.1, speed=5.0, device='cuda:0'):
    times = np.arange(-num_history + 1, 1) * dt
    positions = np.zeros((num_history, 3))
    positions[:, 0] = times * speed
    
    ego_history_xyz = torch.from_numpy(positions).float().unsqueeze(0).unsqueeze(0).to(device)
    ego_history_rot = torch.eye(3).unsqueeze(0).repeat(num_history, 1, 1).unsqueeze(0).unsqueeze(0).to(device)
    return ego_history_xyz, ego_history_rot

def main():
    model_orig, model_tuned = load_models()
    processor = helper.get_processor(model_orig.tokenizer) # Shared processor
    
    # Select samples
    all_images = sorted(FOLDER_PATH.glob("*.jpg"))
    if len(all_images) < NUM_FRAMES:
        print("Not enough images.")
        return
        
    indices = np.linspace(0, len(all_images) - NUM_FRAMES - 1, NUM_SAMPLES, dtype=int)
    
    results = []
    
    print(f"Running inference on {NUM_SAMPLES} samples...")
    
    for i, idx in enumerate(indices):
        frame_paths = [all_images[idx + j] for j in range(NUM_FRAMES)]
        
        # Prepare inputs
        # We need to construct inputs manually for Alpamayo logic
        # Based on test_rellis3d.py
        
        # 1. Images -> Tensor
        frames = []
        for fp in frame_paths:
             frames.append(torch.from_numpy(np.array(Image.open(fp).convert("RGB"))).permute(2,0,1).float()/255.0)
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
        
        # --- Run Model Orig (GPU 0) ---
        hist_xyz_0, hist_rot_0 = get_history_tensors(device="cuda:0")
        # Move inputs to correct device and dtype
        inputs_0 = {k: v.to(device="cuda:0") for k, v in inputs.items()}
        
        # Enable autocast for mixed precision (Validation usually requires matching dtypes)
        # Diffusion often runs in FP32, but model is BF16. Autocast helps.
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                outputs_orig = model_orig.sample_trajectories_from_data_with_vlm_rollout(
                    data={
                         "tokenized_data": inputs_0,
                         "ego_history_xyz": hist_xyz_0,
                         "ego_history_rot": hist_rot_0
                    },
                    num_traj_sets=1,
                    num_traj_samples=1,
                )
                # outputs_orig is (pred_xyz, pred_rot)
                traj_orig = outputs_orig[0] # XYZ
            
        # --- Run Model Tuned (GPU 1) ---
        
        inputs_1 = {k: v.clone().to(device="cuda:1") for k, v in inputs.items()}
        hist_xyz_1, hist_rot_1 = get_history_tensors(device="cuda:1")
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                outputs_tuned = model_tuned.sample_trajectories_from_data_with_vlm_rollout(
                    data={
                         "tokenized_data": inputs_1,
                         "ego_history_xyz": hist_xyz_1,
                         "ego_history_rot": hist_rot_1
                    },
                    num_traj_sets=1,
                    num_traj_samples=1,
                )
                traj_tuned = outputs_tuned[0]
            
        # Store for viz
        results.append({
            "image_path": frame_paths[-1],
            "traj_orig": traj_orig.cpu(), # Move to CPU
            "traj_tuned": traj_tuned.cpu()
        })
        print(f"Sample {i+1}/{NUM_SAMPLES} done.")

    # Visualization
    print("Generating visualization...")
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()
    
    for i, res in enumerate(results):
        ax = axes[i]
        img = Image.open(res["image_path"])
        ax.imshow(img)
        
        pts_orig = project_traj(res["traj_orig"], np.array(img).shape)
        pts_tuned = project_traj(res["traj_tuned"], np.array(img).shape)
        
        if len(pts_orig) > 1:
            xs, ys = zip(*pts_orig)
            ax.plot(xs, ys, 'b-', linewidth=3, label='Original')
            
        if len(pts_tuned) > 1:
            xs, ys = zip(*pts_tuned)
            ax.plot(xs, ys, 'r-', linewidth=3, label='Fine-Tuned')
            
        ax.set_title(f"Sample {i+1}", fontsize=12)
        ax.axis('off')
        
        if i == 0:
            ax.legend(fontsize=12)
            
    plt.tight_layout()
    save_path = OUTPUT_DIR / "comparison_result.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    main()
