#!/usr/bin/env python3
"""
Generate Fine-Tuning Dataset from Rellis-3D Off-Road Images (V2 - Sample Version).

New approach:
- Trajectory: Use critic to evaluate against guidelines
- Language: Use Gemini to generate concise 1-sentence scene descriptions

This creates a small sample dataset for testing.
"""

import sys
import json
import textwrap
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/byounggun/alpamayo/src')

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

print("="*60)
print("SAMPLE DATASET GENERATION V2 (Gemini Descriptions)")
print("="*60)

# Configuration
SAMPLES_PER_FOLDER = 5
NUM_FRAMES = 4  # Sequential frames
RELLIS3D_ROOT = Path("/home/byounggun/alpamayo/Rellis-3D")
OUTPUT_DIR = Path("/home/byounggun/alpamayo/src/alpamayo_r1/alignment/preference_dataset_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

# Process ALL folders
rellis_folders = sorted([f for f in RELLIS3D_ROOT.iterdir() if f.is_dir()])
print(f"Processing {len(rellis_folders)} folders: {[f.name for f in rellis_folders]}")


# Initialize components
print("\n[1/4] Initializing components...")
from alpamayo_r1.alignment.openrouter_critic_v2 import OffRoadDescriber
from alpamayo_r1.alignment.scene_description_guidelines import load_scene_description_guidelines

# Scene description guidelines (for VLM training)
scene_guidelines = load_scene_description_guidelines()
guidelines_prompt = scene_guidelines.to_prompt()
print(f"✓ Scene Description Guidelines: {len(scene_guidelines)} rules for VLM training")

# Scene describer (for language generation ONLY - no trajectory generation)
describer = OffRoadDescriber(model="google/gemini-3-flash-preview", temperature=0.9)
print(f"✓ Scene Describer: Gemini 3 Flash for terrain/hazard descriptions")

# ==============================================================================
# Main Processing
# ==============================================================================
print(f"\n[2/3] Processing {len(rellis_folders)} folders x {SAMPLES_PER_FOLDER} samples = {len(rellis_folders) * SAMPLES_PER_FOLDER} total samples...")

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

            # Load last frame for Gemini processing
            last_frame_path = frame_paths[-1]
            last_frame_np = np.array(Image.open(last_frame_path).convert("RGB"))

            # ==============================================================
            # GENERATE SCENE DESCRIPTION (Terrain + Hazards ONLY)
            # ==============================================================
            scene_desc = describer.describe_scene(images=[last_frame_np], guidelines=guidelines_prompt)
            gemini_description = scene_desc.description

            # Save sample (VLM training only - no trajectory)
            sample = {
                "sample_id": sample_idx,
                "folder": folder.name,
                "frame_paths": [str(fp) for fp in frame_paths],
                "reasoning": gemini_description,  # Gemini terrain/hazard description
            }
            all_samples.append(sample)

            desc_preview = gemini_description[:70] + "..." if len(gemini_description) > 70 else gemini_description
            print(f"    ✓ #{sample_idx}: {desc_preview}")
            total_ok += 1

        except Exception as e:
            print(f"    ✗ #{sample_idx}: ERROR - {str(e)[:50]}")
            total_errors += 1
            continue

# ==============================================================================
# Save Dataset
# ==============================================================================
print(f"\n[3/3] Saving fine-tuning dataset...")

# Save as JSONL
jsonl_path = OUTPUT_DIR / "finetune_data.jsonl"
with open(jsonl_path, "w") as f:
    for sample in all_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# Save summary JSON
summary = {
    "version": "v2_sample",
    "description": "VLM-only training: Gemini terrain/hazard descriptions (no trajectory, no driving actions)",
    "total_samples": len(all_samples),
    "successful_samples": total_ok,
    "errors": total_errors,
    "folders_processed": len(rellis_folders),
    "samples_per_folder": SAMPLES_PER_FOLDER,
    "num_frames": NUM_FRAMES,
    "language_model": "google/gemini-3-flash-preview",
    "training_target": "VLM only (terrain/hazard description)",
    "created_at": datetime.now().isoformat(),
}
with open(OUTPUT_DIR / "dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"✓ Saved {len(all_samples)} samples to {jsonl_path}")

# ==============================================================================
# Summary
# ==============================================================================
print(f"\n[Summary]")
print("="*60)
print("SAMPLE DATASET COMPLETE (V2 - VLM Only)!")
print("="*60)
print(f"Total samples: {len(all_samples)}")
print(f"  Successful: {total_ok}")
print(f"  Errors: {total_errors}")
print(f"\nData type: Terrain/hazard descriptions (NO trajectory, NO driving actions)")
print(f"Model: Gemini 3 Flash")
print(f"Training target: VLM only")
print(f"\nOutput: {OUTPUT_DIR}")
print(f"  - finetune_data.jsonl")
print(f"  - dataset_summary.json")

# ==============================================================================
# Visualize Samples
# ==============================================================================
print(f"\n[4/4] Creating visualization of sample descriptions...")

# Create grid visualization
num_viz = min(20, len(all_samples))  # Visualize up to 20 samples
cols = 4
rows = (num_viz + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
axes = axes.flatten() if num_viz > 1 else [axes]

for i in range(num_viz):
    ax = axes[i]
    sample = all_samples[i]

    # Load last frame image
    img_path = sample["frame_paths"][-1]
    try:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        ax.imshow(img_np)
    except Exception as e:
        print(f"  Error loading image {img_path}: {e}")
        ax.text(0.5, 0.5, "Image Load Error", ha='center', va='center')
        ax.set_title(f"Sample #{sample['sample_id']}", fontsize=10)
        ax.axis('off')
        continue

    # Get reasoning description
    reasoning = sample["reasoning"]

    # Wrap text for better display (max 60 chars per line)
    wrapped_text = textwrap.fill(reasoning, width=60)

    # Truncate if still too long (max 3 lines for title)
    lines = wrapped_text.split('\n')
    if len(lines) > 3:
        display_text = '\n'.join(lines[:3]) + '...'
    else:
        display_text = wrapped_text

    # Set title with sample info and reasoning
    ax.set_title(
        f"Sample #{sample['sample_id']} ({sample['folder']})\n{display_text}",
        fontsize=9,
        color='darkblue',
        wrap=True
    )
    ax.axis('off')

# Hide unused subplots
for j in range(num_viz, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
viz_path = OUTPUT_DIR / "sample_descriptions_viz.png"
plt.savefig(viz_path, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved to {viz_path}")

print("\n" + "="*60)
print("ALL DONE!")
print("="*60)
