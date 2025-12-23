#!/usr/bin/env python3
"""
Inference script to compare Original vs Fine-tuned Alpamayo model.
Both models run in bfloat16 (no quantization needed with sufficient VRAM).
"""

import os
import sys
import copy
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from peft import PeftModel

sys.path.insert(0, '/home/byounggun/alpamayo/src')

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# Paths
CHECKPOINT_PATH = "/home/byounggun/alpamayo/outputs/alpamayo_full_finetuned"
DATA_PATH = "/home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset/finetune_data.jsonl"
DEVICE = "cuda:0"


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
    print("Loading ORIGINAL Alpamayo model (bfloat16)...")
    print("=" * 60)

    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
    )
    model = model.to(DEVICE)

    model.eval()
    print("Original model loaded!")
    return model


def load_finetuned_model():
    """Load fine-tuned model with VLM LoRA + Expert LoRA + Diffusion decoder weights."""
    print("\n" + "=" * 60)
    print("Loading FINE-TUNED Alpamayo model (bfloat16 with full weights)...")
    print("=" * 60)

    # Load base model
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
    )
    model = model.to(DEVICE)

    # Load VLM LoRA
    vlm_lora_path = os.path.join(CHECKPOINT_PATH, "final", "vlm_lora")
    if os.path.exists(vlm_lora_path):
        print(f"Loading VLM LoRA from {vlm_lora_path}...")
        model.vlm = PeftModel.from_pretrained(model.vlm, vlm_lora_path)
        print("  VLM LoRA loaded!")
    else:
        print(f"  Warning: VLM LoRA not found at {vlm_lora_path}, using base VLM")

    # Load Expert LoRA + Diffusion decoder weights
    checkpoint_path = os.path.join(CHECKPOINT_PATH, "final", "expert_diffusion.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading Expert + Diffusion weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        # Load Expert LoRA (if saved as full state dict, we need to apply LoRA first)
        if "expert_lora" in checkpoint:
            # The checkpoint contains full state dict including LoRA weights
            # We need to first apply LoRA config then load the weights
            from peft import LoraConfig, get_peft_model
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
            print("  Expert LoRA loaded!")

        if "action_in_proj" in checkpoint:
            model.action_in_proj.load_state_dict(checkpoint["action_in_proj"])
            print("  action_in_proj loaded!")

        if "action_out_proj" in checkpoint:
            model.action_out_proj.load_state_dict(checkpoint["action_out_proj"])
            print("  action_out_proj loaded!")
    else:
        print(f"  Warning: checkpoint not found at {checkpoint_path}")

    model.eval()
    print("Fine-tuned model loaded successfully!")
    return model


def load_test_sample(data_path, idx=0):
    """Load a test sample from the training data."""
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    return None


def run_inference(model, processor, frame_paths, device=DEVICE):
    """Run inference using the same method as compare_models.py."""
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
    
    # Run inference with autocast
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
    
    return pred_xyz


def print_trajectory(name, pred_xyz, gt_traj=None):
    """Print trajectory comparison."""
    print(f"\n{name} Trajectory (first 10 waypoints):")
    traj = pred_xyz[0, 0, :10, :2].cpu().numpy()
    for i in range(len(traj)):
        x, y = float(traj[i, 0]), float(traj[i, 1])
        gt_str = ""
        if gt_traj and i < len(gt_traj):
            gx, gy = float(gt_traj[i][0]), float(gt_traj[i][1])
            diff = float(np.sqrt((x - gx)**2 + (y - gy)**2))
            gt_str = f"  | GT: ({gx:.2f}, {gy:.2f}) | diff: {diff:.2f}m"
        print(f"  {i+1}: ({x:.3f}, {y:.3f}){gt_str}")


def main():
    print("=" * 60)
    print("COMPARISON: Original vs Fine-tuned Alpamayo")
    print("=" * 60)
    
    # Load test sample first
    print("\nLoading test sample...")
    sample = load_test_sample(DATA_PATH, idx=0)
    if sample is None:
        print("Failed to load test sample")
        return
    
    frame_paths = sample.get("frame_paths", [])[:4]
    gt_traj = sample.get("trajectory", [])
    print(f"Using {len(frame_paths)} frames")
    print(f"Ground truth has {len(gt_traj)} waypoints")
    
    original_pred = None
    finetuned_pred = None
    
    # ===== ORIGINAL MODEL =====
    try:
        original_model = load_original_model()
        processor = helper.get_processor(original_model.tokenizer)

        print("\nRunning inference with ORIGINAL model...")
        original_pred = run_inference(original_model, processor, frame_paths)

        # Free memory
        del original_model
        torch.cuda.empty_cache()
        print("Original model unloaded, memory freed.")
    except Exception as e:
        print(f"Original model inference failed: {e}")
        import traceback
        traceback.print_exc()

    # ===== FINE-TUNED MODEL =====
    try:
        finetuned_model = load_finetuned_model()
        processor = helper.get_processor(finetuned_model.tokenizer)

        print("\nRunning inference with FINE-TUNED model...")
        finetuned_pred = run_inference(finetuned_model, processor, frame_paths, device=DEVICE)

        # Free memory
        del finetuned_model
        torch.cuda.empty_cache()
        print("Fine-tuned model unloaded, memory freed.")
    except Exception as e:
        print(f"Fine-tuned model inference failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ===== RESULTS =====
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    if gt_traj:
        print(f"\nGround Truth (first 10 points):")
        for i, (x, y) in enumerate(gt_traj[:10]):
            print(f"  {i+1}: ({x:.3f}, {y:.3f})")
    
    if original_pred is not None:
        print_trajectory("ORIGINAL", original_pred, gt_traj)
    else:
        print("\nORIGINAL: Failed to get prediction")
    
    if finetuned_pred is not None:
        print_trajectory("FINE-TUNED", finetuned_pred, gt_traj)
    else:
        print("\nFINE-TUNED: Failed to get prediction")
    
    # Calculate average distance to GT
    if original_pred is not None and finetuned_pred is not None and gt_traj:
        n = min(len(gt_traj), original_pred.shape[2], finetuned_pred.shape[2])
        orig_traj = original_pred[0, 0, :n, :2].cpu().numpy()
        ft_traj = finetuned_pred[0, 0, :n, :2].cpu().numpy()
        gt_arr = np.array(gt_traj[:n])
        
        orig_dist = np.mean(np.sqrt(np.sum((orig_traj - gt_arr)**2, axis=1)))
        ft_dist = np.mean(np.sqrt(np.sum((ft_traj - gt_arr)**2, axis=1)))
        
        print(f"\n" + "=" * 60)
        print(f"AVERAGE DISTANCE TO GROUND TRUTH:")
        print(f"  Original:   {orig_dist:.3f} m")
        print(f"  Fine-tuned: {ft_dist:.3f} m")
        improvement = orig_dist - ft_dist
        if orig_dist > 0:
            print(f"  Improvement: {improvement:.3f} m ({improvement / orig_dist * 100:.1f}%)")
        print("=" * 60)
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
