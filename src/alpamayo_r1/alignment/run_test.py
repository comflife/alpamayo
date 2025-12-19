#!/usr/bin/env python3
"""
End-to-end test script for the alignment pipeline.
Loads a scene, runs Alpamayo inference, gets GPT critique, and saves results.
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

# Ensure alignment module can find .env
sys.path.insert(0, '/home/byounggun/alpamayo/src')

import numpy as np
import torch
from PIL import Image

# First, test OpenRouter connection before loading heavy model
print("="*60)
print("ALIGNMENT PIPELINE TEST")
print("="*60)

print("\n[1/5] Testing OpenRouter API connection...")
from alpamayo_r1.alignment.openrouter_critic import OpenRouterCritic
from alpamayo_r1.alignment.guidelines import load_guidelines

try:
    critic = OpenRouterCritic(model="openai/gpt-4o-mini")
    test_result = critic.test_connection()
    if test_result["success"]:
        print(f"✓ API connected! Response: {test_result['response']}")
    else:
        print(f"✗ API failed: {test_result['error']}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Failed to initialize critic: {e}")
    sys.exit(1)

print("\n[2/5] Loading guidelines...")
guidelines = load_guidelines()
print(f"✓ Loaded {len(guidelines)} rules")
print(f"  Guidelines prompt:\n{guidelines.to_prompt()}")

print("\n[3/5] Loading Alpamayo-R1 model (this may take a while)...")
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

# Use a default clip_id
clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
t0_us = 5_100_000

print(f"  Loading model from nvidia/Alpamayo-R1-10B...")
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)
print("✓ Model loaded!")

print(f"\n[4/5] Loading scene data (clip_id: {clip_id})...")
data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
print(f"✓ Data loaded! Image shape: {data['image_frames'].shape}")

# Prepare inputs
images = data["image_frames"].flatten(0, 1)  # (N_cameras * num_frames, C, H, W)
messages = helper.create_message(images)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)
model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}
model_inputs = helper.to_device(model_inputs, "cuda")

print("\n  Running Alpamayo inference...")
torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,
        max_generation_length=256,
        return_extra=True,
    )

reasoning = extra["cot"][0, 0, 0] if extra.get("cot") is not None else "No reasoning available"
trajectory = pred_xyz.cpu().numpy()[0, 0, 0]  # (T, 3)

print("✓ Inference complete!")
print(f"\n  Chain-of-Causation Reasoning:")
print(f"  \"{reasoning[:200]}...\"" if len(reasoning) > 200 else f"  \"{reasoning}\"")
print(f"\n  Trajectory: {trajectory.shape[0]} waypoints")
print(f"  Start: ({trajectory[0, 0]:.2f}, {trajectory[0, 1]:.2f})")
print(f"  End: ({trajectory[-1, 0]:.2f}, {trajectory[-1, 1]:.2f})")
print(f"  Max lateral deviation: {np.abs(trajectory[:, 1]).max():.2f}m")

print("\n[5/5] Sending to GPT for critique...")

# Prepare images for GPT (take 4 representative frames)
images_np = images[::4][:4].permute(0, 2, 3, 1).cpu().numpy()  # (4, H, W, C)

# Get critique
critique = critic.critique(
    images=list(images_np),
    reasoning=reasoning,
    trajectory=trajectory[:, :2],  # xy only
    guidelines=guidelines.to_prompt(),
)

print("✓ Critique received!")
print(f"\n  Violated: {critique.violated}")
print(f"  Explanation: {critique.explanation}")
if critique.corrected_reasoning:
    print(f"\n  Corrected Reasoning:")
    print(f"  \"{critique.corrected_reasoning}\"")

# Save results
output_dir = Path("/home/byounggun/alpamayo/src/alpamayo_r1/alignment/test_output")
output_dir.mkdir(exist_ok=True)

# Save images
for i, img in enumerate(images_np):
    Image.fromarray(img.astype(np.uint8)).save(output_dir / f"image_{i}.jpg")

# Save result JSON
result = {
    "timestamp": datetime.now().isoformat(),
    "clip_id": clip_id,
    "t0_us": t0_us,
    "alpamayo_output": {
        "reasoning": reasoning,
        "trajectory_first_10": trajectory[:10].tolist(),
        "trajectory_last_5": trajectory[-5:].tolist(),
    },
    "critique": critique.to_dict(),
    "guidelines": guidelines.to_prompt(),
}

result_path = output_dir / "test_result.json"
with open(result_path, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n" + "="*60)
print("TEST COMPLETE!")
print("="*60)
print(f"Results saved to: {output_dir}")
print(f"  - test_result.json")
print(f"  - image_0.jpg ~ image_3.jpg")
