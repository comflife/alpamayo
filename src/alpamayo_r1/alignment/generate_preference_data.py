#!/usr/bin/env python3
"""
Generate Fine-Tuning Dataset from Rellis-3D Off-Road Images.

Output format for each sample:
- If OK (no correction): Alpamayo's reasoning + trajectory
- If VIOLATED (corrected): GPT's corrected reasoning + trajectory

This creates training data for LoRA fine-tuning.
"""

import sys
import json
import copy
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/byounggun/alpamayo/src')

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

print("="*60)
print("FINE-TUNING DATASET GENERATION (Rellis-3D)")
print("="*60)

# Configuration
SAMPLES_PER_FOLDER = 1000  # More samples for better coverage
NUM_FRAMES = 4  # Sequential frames for Alpamayo
RELLIS3D_ROOT = Path("/home/byounggun/alpamayo/Rellis-3D")
OUTPUT_DIR = Path("/home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# Use only 00000 folder for now
rellis_folders = [RELLIS3D_ROOT / "00000"]
print(f"Using folder: {rellis_folders[0]}")

# Camera intrinsics
FX, FY = 2813.64, 2808.33
CX, CY = 969.29, 624.05

# Initialize components
print("\n[1/4] Initializing components...")
from alpamayo_r1.alignment.openrouter_critic import OpenRouterCritic
from alpamayo_r1.alignment.guidelines import load_guidelines
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

critic = OpenRouterCritic(model="google/gemini-3-flash-preview")
guidelines = load_guidelines()
guidelines_prompt = guidelines.to_prompt()
print(f"✓ Critic: Gemini 3 Flash")
print(f"✓ Guidelines: {len(guidelines)} off-road rules")

print("\n  Loading Alpamayo-R1 model...")
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)
print("✓ Model loaded!")

# ==============================================================================
# Helper Functions
# ==============================================================================
def create_ego_history(device):
    """Create realistic straight-line ego history for Alpamayo."""
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

# ==============================================================================
# Main Processing
# ==============================================================================
print(f"\n[2/4] Processing {len(rellis_folders)} folders x {SAMPLES_PER_FOLDER} samples = {len(rellis_folders) * SAMPLES_PER_FOLDER} total samples...")

all_samples = []
total_ok = 0
total_violated = 0
total_errors = 0
sample_idx = 0

for folder_idx, folder in enumerate(rellis_folders):
    image_dir = folder / "pylon_camera_node"
    if not image_dir.exists():
        print(f"  Skipping {folder.name}: no pylon_camera_node")
        continue
    
    all_images = sorted(image_dir.glob("*.jpg"))
    if len(all_images) < NUM_FRAMES:
        print(f"  Skipping {folder.name}: not enough images")
        continue
    
    print(f"\n  [{folder_idx+1}/{len(rellis_folders)}] {folder.name}: {len(all_images)} images")
    
    # Sample positions (spaced out)
    sample_starts = np.linspace(0, len(all_images) - NUM_FRAMES - 1, SAMPLES_PER_FOLDER, dtype=int)
    
    for local_idx, start_idx in enumerate(sample_starts):
        sample_idx += 1
        
        try:
            # Load 4 sequential frames
            frame_paths = [all_images[start_idx + j] for j in range(NUM_FRAMES)]
            frames = []
            for fp in frame_paths:
                img = Image.open(fp).convert("RGB")
                img_np = np.array(img)
                img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                frames.append(img_t)
            
            frames_tensor = torch.stack(frames, dim=0)
            
            # Create message for Alpamayo
            messages = helper.create_message(frames_tensor)
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            ego_history_xyz, ego_history_rot = create_ego_history("cuda")
            
            model_inputs = {
                "tokenized_data": inputs,
                "ego_history_xyz": ego_history_xyz,
                "ego_history_rot": ego_history_rot,
            }
            model_inputs = helper.to_device(model_inputs, "cuda")
            
            # Run Alpamayo inference
            torch.cuda.manual_seed_all(42 + sample_idx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=copy.deepcopy(model_inputs),
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=256,
                    return_extra=True,
                )
            
            alpamayo_trajectory = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]
            alpamayo_reasoning = extra["cot"][0, 0, 0] if extra.get("cot") is not None else ""
            
            # Get last frame for critique
            last_frame_np = np.array(Image.open(frame_paths[-1]))
            
            # Critique
            critique = critic.critique(
                images=[last_frame_np],
                reasoning=alpamayo_reasoning,
                trajectory=alpamayo_trajectory,
                guidelines=guidelines_prompt,
            )
            
            # ==============================================================
            # CREATE FINE-TUNING SAMPLE
            # ==============================================================
            if critique.violated and critique.corrected_reasoning and critique.corrected_trajectory is not None:
                # Use CORRECTED data
                final_reasoning = critique.corrected_reasoning
                final_trajectory = critique.corrected_trajectory.tolist()
                source = "corrected"
                total_violated += 1
            else:
                # Use ORIGINAL Alpamayo data
                final_reasoning = alpamayo_reasoning
                final_trajectory = alpamayo_trajectory.tolist()
                source = "alpamayo"
                total_ok += 1
            
            # Save sample
            sample = {
                "sample_id": sample_idx,
                "folder": folder.name,
                "frame_paths": [str(fp) for fp in frame_paths],
                "reasoning": final_reasoning,
                "trajectory": final_trajectory,
                "source": source,  # "alpamayo" or "corrected"
                # Also save original for reference
                "original_reasoning": alpamayo_reasoning,
                "original_trajectory": alpamayo_trajectory.tolist(),
                "violated": critique.violated,
                "explanation": critique.explanation or "",
            }
            all_samples.append(sample)
            
            status = "✓" if source == "alpamayo" else "★"
            print(f"    {status} #{sample_idx}: {source} - {(critique.explanation or 'OK')[:40]}...")
            
        except Exception as e:
            print(f"    ✗ #{sample_idx}: ERROR - {str(e)[:50]}")
            total_errors += 1
            continue

# ==============================================================================
# Save Dataset
# ==============================================================================
print(f"\n[3/4] Saving fine-tuning dataset...")

# Save as JSONL
jsonl_path = OUTPUT_DIR / "finetune_data.jsonl"
with open(jsonl_path, "w") as f:
    for sample in all_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# Save summary JSON
summary = {
    "total_samples": len(all_samples),
    "alpamayo_samples": total_ok,
    "corrected_samples": total_violated,
    "errors": total_errors,
    "folders_processed": len(rellis_folders),
    "samples_per_folder": SAMPLES_PER_FOLDER,
    "num_frames": NUM_FRAMES,
    "created_at": datetime.now().isoformat(),
}
with open(OUTPUT_DIR / "dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"✓ Saved {len(all_samples)} samples to {jsonl_path}")

# ==============================================================================
# Summary
# ==============================================================================
print(f"\n[4/4] Summary")
print("="*60)
print("FINE-TUNING DATASET COMPLETE!")
print("="*60)
print(f"Total samples: {len(all_samples)}")
print(f"  Alpamayo (OK): {total_ok}")
print(f"  Corrected (VIOLATED): {total_violated}")
print(f"  Errors: {total_errors}")
print(f"\nOutput: {OUTPUT_DIR}")
print(f"  - finetune_data.jsonl")
print(f"  - dataset_summary.json")
