#!/usr/bin/env python3
"""
Test RLAIF Evaluator with a real sample from the dataset.
"""

import sys
import json
import numpy as np
from PIL import Image
from pathlib import Path

sys.path.insert(0, '/home/byounggun/alpamayo/src')

from alpamayo_r1.alignment.rlaif_evaluator import RLAIFEvaluator
from alpamayo_r1.alignment.guidelines import load_guidelines

print("=" * 60)
print("RLAIF EVALUATOR TEST")
print("=" * 60)

# Load guidelines
print("\n[1/4] Loading guidelines...")
guidelines = load_guidelines()
guidelines_text = guidelines.to_prompt()
print(f"✓ Loaded {len(guidelines)} guidelines")

# Initialize evaluator
print("\n[2/4] Initializing RLAIF evaluator...")
evaluator = RLAIFEvaluator(
    model="google/gemini-flash-1.5",
    temperature=0.3,
)
print("✓ Evaluator initialized")

# Load a sample from dataset
print("\n[3/4] Loading test sample...")
data_path = Path("/home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset/finetune_data.jsonl")

with open(data_path, 'r') as f:
    # Get first sample
    sample = json.loads(f.readline())

print(f"Sample ID: {sample['sample_id']}")
print(f"Folder: {sample['folder']}")
print(f"Reasoning: {sample['reasoning'][:100]}...")

# Load image (last frame)
img_path = sample['frame_paths'][-1]
img = np.array(Image.open(img_path).convert("RGB"))
print(f"Image loaded: {img.shape}")

# Create test candidates
# Candidate 1: Original from dataset
gt_traj = np.array(sample['trajectory'])
gt_reasoning = sample['reasoning']

# Candidate 2: Modified trajectory (shift left)
left_traj = gt_traj.copy()
left_traj[:, 1] += 1.5  # Shift 1.5m to the left
left_reasoning = "Turn left to avoid obstacle on the right"

# Candidate 3: Modified trajectory (shift right)
right_traj = gt_traj.copy()
right_traj[:, 1] -= 1.5  # Shift 1.5m to the right
right_reasoning = "Turn right to follow the clear path on the right side"

# Candidate 4: Straight trajectory
straight_traj = gt_traj.copy()
straight_traj[:, 1] *= 0.2  # Reduce lateral movement
straight_reasoning = "Continue straight as the path ahead is clear"

trajectories = [gt_traj, left_traj, right_traj, straight_traj]
reasonings = [gt_reasoning, left_reasoning, right_reasoning, straight_reasoning]

print(f"\nCreated {len(trajectories)} test candidates")

# Evaluate
print("\n[4/4] Evaluating candidates with LLM...")
print("(This may take 10-30 seconds...)")

try:
    scores = evaluator.evaluate_candidates(
        image=img,
        trajectories=trajectories,
        reasonings=reasonings,
        guidelines=guidelines_text,
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for i, (score, reasoning) in enumerate(zip(scores, reasonings), 1):
        print(f"\n--- Candidate {i} ---")
        print(f"Reasoning: {reasoning[:60]}...")
        print(f"Trajectory Quality: {score.trajectory_quality:.1f}/10")
        print(f"Reasoning Quality:  {score.reasoning_quality:.1f}/10")
        print(f"Consistency:        {score.consistency:.1f}/10")
        print(f"Overall Reward:     {score.overall_reward:.2f}/10")
        print(f"Explanation: {score.explanation}")

    # Find best candidate
    best_idx = max(range(len(scores)), key=lambda i: scores[i].overall_reward)
    print("\n" + "=" * 60)
    print(f"BEST CANDIDATE: #{best_idx + 1}")
    print(f"Overall Reward: {scores[best_idx].overall_reward:.2f}/10")
    print("=" * 60)

    print("\n✓ Test completed successfully!")

except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
