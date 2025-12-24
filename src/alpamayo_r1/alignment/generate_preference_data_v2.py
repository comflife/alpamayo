#!/usr/bin/env python3
"""
Generate Fine-Tuning Dataset from Rellis-3D Off-Road Images (V2).

New approach:
- Trajectory: Use critic to evaluate against guidelines (same as before)
- Language: Use Gemini to generate detailed scene descriptions (no Alpamayo inference)

Output format for each sample:
- Trajectory: Corrected by critic if violated, otherwise original GT
- Language: Detailed scene description from Gemini

Features:
- Incremental saving: Results saved after each sample
- Per-folder output: Each folder has its own JSONL file
- Resume support: Automatically skips already processed samples
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/byounggun/alpamayo/src')

import numpy as np
from PIL import Image

print("="*60)
print("FINE-TUNING DATASET GENERATION (Rellis-3D)")
print("="*60)

# Configuration
SAMPLES_PER_FOLDER = 500
NUM_FRAMES = 4  # Sequential frames
RELLIS3D_ROOT = Path("/home/byounggun/alpamayo/Rellis-3D")
OUTPUT_DIR = Path("/home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

# Process ALL folders
rellis_folders = sorted([f for f in RELLIS3D_ROOT.iterdir() if f.is_dir()])
print(f"Processing {len(rellis_folders)} folders: {[f.name for f in rellis_folders]}")


# ==============================================================================
# Resume Support Functions
# ==============================================================================
def load_processed_samples(folder_name: str) -> set:
    """Load already processed sample indices for a folder."""
    folder_output = OUTPUT_DIR / f"{folder_name}.jsonl"
    processed = set()
    if folder_output.exists():
        with open(folder_output, "r") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    # Use local_idx to track which samples are done
                    if "local_idx" in sample:
                        processed.add(sample["local_idx"])
                except json.JSONDecodeError:
                    continue
    return processed


def save_sample_to_folder(folder_name: str, sample: dict):
    """Append a single sample to the folder's JSONL file."""
    folder_output = OUTPUT_DIR / f"{folder_name}.jsonl"
    with open(folder_output, "a") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_folder_stats(folder_name: str) -> dict:
    """Load stats for already processed samples in a folder."""
    folder_output = OUTPUT_DIR / f"{folder_name}.jsonl"
    stats = {"ok": 0, "violated": 0, "total": 0}
    if folder_output.exists():
        with open(folder_output, "r") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    stats["total"] += 1
                    if sample.get("source") == "alpamayo":
                        stats["ok"] += 1
                    else:
                        stats["violated"] += 1
                except json.JSONDecodeError:
                    continue
    return stats


# Initialize components
print("\n[1/3] Initializing Gemini describer...")
from alpamayo_r1.alignment.openrouter_critic_v2 import OffRoadDescriber
from alpamayo_r1.alignment.scene_description_guidelines import load_scene_description_guidelines

# Scene description guidelines (for VLM training)
scene_guidelines = load_scene_description_guidelines()
guidelines_prompt = scene_guidelines.to_prompt()
print(f"✓ Scene Description Guidelines: {len(scene_guidelines)} rules for VLM training")

# Scene describer (for language generation ONLY - VLM training)
describer = OffRoadDescriber(model="google/gemini-3-flash-preview", temperature=0.9)
print(f"✓ Scene Describer: Gemini 3 Flash for terrain/hazard descriptions (VLM training only)")

# ==============================================================================
# Main Processing
# ==============================================================================
print(f"\n[2/3] Processing {len(rellis_folders)} folders x {SAMPLES_PER_FOLDER} samples = {len(rellis_folders) * SAMPLES_PER_FOLDER} total samples...")

total_ok = 0
total_violated = 0
total_errors = 0
total_skipped = 0
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
    
    # Load already processed samples for this folder (resume support)
    processed_samples = load_processed_samples(folder.name)
    folder_stats = load_folder_stats(folder.name)
    
    if len(processed_samples) > 0:
        print(f"\n  [{folder_idx+1}/{len(rellis_folders)}] {folder.name}: Resuming from {len(processed_samples)} processed samples")
        total_ok += folder_stats["ok"]
        total_violated += folder_stats["violated"]
    else:
        print(f"\n  [{folder_idx+1}/{len(rellis_folders)}] {folder.name}: {len(all_images)} images (starting fresh)")
    
    # Sample positions (spaced out)
    sample_starts = np.linspace(0, len(all_images) - NUM_FRAMES - 1, SAMPLES_PER_FOLDER, dtype=int)
    
    for local_idx, start_idx in enumerate(sample_starts):
        sample_idx += 1
        
        # Skip already processed samples (resume support)
        if local_idx in processed_samples:
            total_skipped += 1
            continue
        
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

            # Create sample with local_idx for resume support (VLM training only)
            sample = {
                "sample_id": sample_idx,
                "local_idx": local_idx,  # For resume tracking
                "folder": folder.name,
                "frame_paths": [str(fp) for fp in frame_paths],
                "reasoning": gemini_description,  # Gemini terrain/hazard description
            }

            # Save immediately to folder-specific file (incremental save)
            save_sample_to_folder(folder.name, sample)

            desc_preview = gemini_description[:70] + "..." if len(gemini_description) > 70 else gemini_description
            print(f"    ✓ #{sample_idx} (local:{local_idx}): {desc_preview}")
            total_ok += 1

        except Exception as e:
            print(f"    ✗ #{sample_idx} (local:{local_idx}): ERROR - {str(e)[:50]}")
            total_errors += 1
            continue

# ==============================================================================
# Save Combined Dataset
# ==============================================================================
print(f"\n[3/3] Combining folder datasets...")

# Combine all folder JSONL files into one
all_samples = []
for folder in rellis_folders:
    folder_output = OUTPUT_DIR / f"{folder.name}.jsonl"
    if folder_output.exists():
        with open(folder_output, "r") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    all_samples.append(sample)
                except json.JSONDecodeError:
                    continue

# Save combined JSONL
jsonl_path = OUTPUT_DIR / "finetune_data.jsonl"
with open(jsonl_path, "w") as f:
    for sample in all_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# Save summary JSON
summary = {
    "version": "v2",
    "description": "VLM-only training: Gemini terrain/hazard descriptions (no trajectory, no driving actions)",
    "total_samples": len(all_samples),
    "successful_samples": total_ok,
    "errors": total_errors,
    "skipped_resumed": total_skipped,
    "folders_processed": len(rellis_folders),
    "samples_per_folder": SAMPLES_PER_FOLDER,
    "num_frames": NUM_FRAMES,
    "language_model": "google/gemini-3-flash-preview",
    "training_target": "VLM only (terrain/hazard description)",
    "created_at": datetime.now().isoformat(),
    "folder_files": [f"{f.name}.jsonl" for f in rellis_folders if (OUTPUT_DIR / f"{f.name}.jsonl").exists()],
}
with open(OUTPUT_DIR / "dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"✓ Combined {len(all_samples)} samples to {jsonl_path}")

# ==============================================================================
# Summary
# ==============================================================================
print(f"\n[Summary]")
print("="*60)
print("FINE-TUNING DATASET COMPLETE (V2 - VLM Only)!")
print("="*60)
print(f"Total samples: {len(all_samples)}")
print(f"  Successful: {total_ok}")
print(f"  Errors: {total_errors}")
print(f"  Skipped (resumed): {total_skipped}")
print(f"\nData type: Terrain/hazard descriptions (NO trajectory, NO driving actions)")
print(f"Model: Gemini 3 Flash")
print(f"Training target: VLM only")
print(f"\nOutput: {OUTPUT_DIR}")
print(f"  - finetune_data.jsonl (combined)")
print(f"  - <folder_name>.jsonl (per folder)")
print(f"  - dataset_summary.json")
