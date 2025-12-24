#!/usr/bin/env python3
"""
RLAIF (Reinforcement Learning from AI Feedback) Trainer for Alpamayo

Uses Claude API to evaluate trajectory + reasoning quality based on guidelines.
Improves both trajectory planning and language consistency without GT dependency.
"""

import os
import sys
import json
import copy
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from anthropic import Anthropic

sys.path.insert(0, '/home/byounggun/alpamayo/src')

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from alpamayo_r1.alignment.guidelines import load_guidelines


@dataclass
class RLAIFConfig:
    """Configuration for RLAIF training."""

    # Model & Data
    model_checkpoint: str = "/home/byounggun/alpamayo/outputs/alpamayo_srd_rl_basic/final"
    data_path: str = "/home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset/finetune_data.jsonl"
    rellis_dir: str = "/home/byounggun/alpamayo/Rellis-3D"

    # LLM Judge
    anthropic_api_key: Optional[str] = None  # Will use env var ANTHROPIC_API_KEY
    judge_model: str = "claude-3-5-sonnet-20241022"

    # Sampling
    num_samples_per_input: int = 4  # Generate N candidates per input
    temperature: float = 0.8
    top_p: float = 0.95

    # Training
    batch_size: int = 2
    num_iterations: int = 100
    save_frequency: int = 10

    # Output
    output_dir: str = "/home/byounggun/alpamayo/outputs/alpamayo_rlaif"
    viz_dir: str = "/home/byounggun/alpamayo/outputs/alpamayo_rlaif/visualizations"

    # Device
    device: str = "cuda:0"


class RLAIFVisualizer:
    """Visualize trajectories on camera images for LLM evaluation."""

    def __init__(self, rellis_dir: str):
        self.rellis_dir = rellis_dir

    def load_camera_params(self, folder: str) -> Tuple[float, float, float, float]:
        """Load camera intrinsics."""
        camera_info_path = os.path.join(self.rellis_dir, f"camera_info_{folder}.txt")
        if os.path.exists(camera_info_path):
            with open(camera_info_path, 'r') as f:
                fx, fy, cx, cy = map(float, f.read().strip().split())
            return fx, fy, cx, cy
        return 2813.64, 2808.33, 969.29, 624.05

    def project_to_image(self, trajectory_xy, img_shape, fx, fy, cx, cy):
        """Project BEV trajectory to image coordinates."""
        H, W = img_shape[:2]
        points = []
        for x_fwd, y_lat in trajectory_xy:
            if x_fwd <= 1.0:
                continue
            u = cx - fx * y_lat / x_fwd
            v = cy + fy * 1.5 / x_fwd  # 1.5m camera height
            if 0 <= u < W and 0 <= v < H:
                points.append((int(u), int(v)))
        return points

    def create_comparison_image(
        self,
        img: np.ndarray,
        trajectories: List[np.ndarray],
        reasonings: List[str],
        folder: str,
        colors: List[str] = None,
        labels: List[str] = None,
    ) -> Image.Image:
        """Create visualization with multiple trajectory candidates."""
        if colors is None:
            colors = ['red', 'blue', 'green', 'orange', 'purple']
        if labels is None:
            labels = [f"Candidate {i+1}" for i in range(len(trajectories))]

        fx, fy, cx, cy = self.load_camera_params(folder)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)

        # Plot all trajectories
        for i, (traj, color, label) in enumerate(zip(trajectories, colors, labels)):
            pts = self.project_to_image(traj, img.shape, fx, fy, cx, cy)
            if len(pts) > 1:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=color, linewidth=4, alpha=0.8, label=label)

        ax.legend(loc='lower right', fontsize=10)
        ax.axis('off')
        ax.set_title('Trajectory Candidates for LLM Evaluation', fontsize=14, fontweight='bold')

        # Convert to PIL Image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return Image.open(buf)


class LLMJudge:
    """Uses Claude API to evaluate trajectory + reasoning quality."""

    def __init__(self, config: RLAIFConfig):
        api_key = config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in config or environment")

        self.client = Anthropic(api_key=api_key)
        self.model = config.judge_model

        # Load guidelines
        self.guidelines = load_guidelines()
        self.guidelines_text = self.guidelines.to_prompt()

    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64."""
        buf = BytesIO()
        image.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def build_evaluation_prompt(
        self,
        trajectories: List[np.ndarray],
        reasonings: List[str],
    ) -> str:
        """Build prompt for LLM judge."""
        prompt = f"""You are an expert off-road autonomous driving evaluator.

# GUIDELINES
{self.guidelines_text}

# YOUR TASK
You are shown a camera image with {len(trajectories)} different trajectory candidates overlaid.
Each candidate has:
- A trajectory path (shown as a colored line)
- A reasoning explanation (text describing why this path was chosen)

For each candidate, evaluate:
1. **Trajectory Quality (0-10)**: How well does the trajectory follow the guidelines above?
   - Does it follow visible tire tracks/paths?
   - Does it avoid puddles, deep mud, ruts, and obstacles?
   - Is it smooth and consistent?
   - Does it maintain safe clearance from edges and vegetation?

2. **Reasoning Quality (0-10)**: How accurately does the reasoning describe the scene and decision?
   - Does it correctly identify terrain features (puddles, paths, obstacles)?
   - Does it explain the trajectory choice clearly?
   - Is it specific and actionable?

3. **Consistency (0-10)**: Do the trajectory and reasoning match?
   - If reasoning says "turn right", does the trajectory turn right?
   - If reasoning mentions avoiding a puddle, does the trajectory actually avoid it?

# CANDIDATES

"""

        for i, (traj, reasoning) in enumerate(zip(trajectories, reasonings)):
            # Compute trajectory stats
            lateral_range = np.abs(traj[:, 1]).max() if len(traj) > 0 else 0
            direction = "left" if traj[-1, 1] < -0.5 else "right" if traj[-1, 1] > 0.5 else "straight"

            prompt += f"""
## Candidate {i+1}
- **Reasoning**: {reasoning}
- **Trajectory Stats**: Direction={direction}, Max lateral offset={lateral_range:.2f}m

"""

        prompt += """
# OUTPUT FORMAT
Return a JSON object with this structure:
{
  "evaluations": [
    {
      "candidate": 1,
      "trajectory_quality": <0-10>,
      "reasoning_quality": <0-10>,
      "consistency": <0-10>,
      "explanation": "<brief explanation of scores>"
    },
    ...
  ],
  "best_candidate": <1-N>,
  "best_candidate_rationale": "<why this candidate is best overall>"
}

Be critical and precise. Use the full 0-10 scale.
"""
        return prompt

    def evaluate_candidates(
        self,
        image: Image.Image,
        trajectories: List[np.ndarray],
        reasonings: List[str],
    ) -> Dict:
        """Send to Claude API and get evaluation."""
        prompt = self.build_evaluation_prompt(trajectories, reasonings)

        # Convert image to base64
        img_b64 = self.image_to_base64(image)

        # Call Claude API
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        # Parse response
        response_text = message.content[0].text

        # Extract JSON from response (might be wrapped in markdown)
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        evaluation = json.loads(json_str)
        return evaluation


class RLAIFTrainer:
    """Main RLAIF training loop."""

    def __init__(self, config: RLAIFConfig):
        self.config = config

        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.viz_dir, exist_ok=True)

        # Load model
        print(f"Loading model from {config.model_checkpoint}...")
        self.model = self._load_model()
        self.processor = helper.get_processor(self.model.tokenizer)

        # Initialize components
        self.visualizer = RLAIFVisualizer(config.rellis_dir)
        self.judge = LLMJudge(config)

        # Load dataset
        print(f"Loading dataset from {config.data_path}...")
        with open(config.data_path, 'r') as f:
            self.dataset = [json.loads(line) for line in f]
        print(f"Loaded {len(self.dataset)} samples")

        # Training state
        self.iteration = 0
        self.reward_history = []

    def _load_model(self) -> AlpamayoR1:
        """Load the Alpamayo model with LoRA adapters."""
        # TODO: Implement model loading similar to run_comparison_v2.py
        # For now, placeholder
        raise NotImplementedError("Model loading to be implemented")

    def create_ego_history(self):
        """Create ego history for inference."""
        num_history = 16
        dt = 0.1
        speed = 5.0
        times = np.arange(-num_history + 1, 1) * dt
        positions = np.zeros((num_history, 3))
        positions[:, 0] = times * speed

        ego_history_xyz = torch.from_numpy(positions).float().unsqueeze(0).unsqueeze(0).to(self.config.device)
        ego_history_rot = torch.eye(3).unsqueeze(0).repeat(num_history, 1, 1).unsqueeze(0).unsqueeze(0).to(self.config.device)

        return ego_history_xyz, ego_history_rot

    def sample_candidates(
        self,
        frame_paths: List[str],
        num_samples: int,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Sample multiple trajectory + reasoning candidates."""
        # Load frames
        frames = []
        for fp in frame_paths:
            if os.path.exists(fp):
                img = Image.open(fp).convert("RGB")
                img_np = np.array(img)
                img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                frames.append(img_t)

        if not frames:
            return [], []

        frames_tensor = torch.stack(frames, dim=0)
        messages = helper.create_message(frames_tensor)
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt",
        )

        ego_history_xyz, ego_history_rot = self.create_ego_history()

        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        model_inputs = helper.to_device(model_inputs, self.config.device)

        # Sample multiple candidates
        trajectories = []
        reasonings = []

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(num_samples):
                    pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                        data=copy.deepcopy(model_inputs),
                        top_p=self.config.top_p,
                        temperature=self.config.temperature,
                        num_traj_samples=1,
                        max_generation_length=256,
                        return_extra=True,
                    )

                    # Extract trajectory and reasoning
                    traj = pred_xyz[0, 0, 0, :, :2].cpu().numpy()
                    reasoning = extra["cot"][0, 0, 0] if extra.get("cot") is not None else "No reasoning provided"

                    trajectories.append(traj)
                    reasonings.append(reasoning)

        return trajectories, reasonings

    def train_step(self, sample: Dict) -> Dict:
        """Execute one RLAIF training step."""
        # Extract sample data
        frame_paths = sample['frame_paths'][:4]
        folder = sample['folder']

        # Load last frame for visualization
        img_path = frame_paths[-1]
        img = np.array(Image.open(img_path).convert("RGB"))

        # Generate candidates
        print(f"  Sampling {self.config.num_samples_per_input} candidates...")
        trajectories, reasonings = self.sample_candidates(
            frame_paths, self.config.num_samples_per_input
        )

        if not trajectories:
            return {"error": "Failed to generate candidates"}

        # Create visualization
        viz_image = self.visualizer.create_comparison_image(
            img, trajectories, reasonings, folder
        )

        # Get LLM evaluation
        print(f"  Querying LLM judge...")
        evaluation = self.judge.evaluate_candidates(viz_image, trajectories, reasonings)

        # Compute rewards
        rewards = []
        for eval_item in evaluation['evaluations']:
            # Average of three scores
            reward = (
                eval_item['trajectory_quality'] +
                eval_item['reasoning_quality'] +
                eval_item['consistency']
            ) / 3.0
            rewards.append(reward)

        # TODO: Use rewards to update model (PPO/DPO style)
        # For now, just log

        return {
            "rewards": rewards,
            "best_candidate": evaluation['best_candidate'],
            "evaluation": evaluation,
            "viz_image": viz_image,
        }

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("STARTING RLAIF TRAINING")
        print("=" * 60)

        for iteration in range(self.config.num_iterations):
            self.iteration = iteration
            print(f"\n[Iteration {iteration+1}/{self.config.num_iterations}]")

            # Sample random batch
            batch_indices = np.random.choice(len(self.dataset), self.config.batch_size, replace=False)
            batch_samples = [self.dataset[i] for i in batch_indices]

            batch_rewards = []

            for i, sample in enumerate(batch_samples):
                print(f"\n  Sample {i+1}/{self.config.batch_size}")
                result = self.train_step(sample)

                if "error" in result:
                    print(f"    Error: {result['error']}")
                    continue

                batch_rewards.extend(result['rewards'])

                # Save visualization
                if iteration % self.config.save_frequency == 0:
                    viz_path = os.path.join(
                        self.config.viz_dir,
                        f"iter_{iteration:04d}_sample_{i:02d}.png"
                    )
                    result['viz_image'].save(viz_path)
                    print(f"    Saved visualization to {viz_path}")

                print(f"    Rewards: {result['rewards']}")
                print(f"    Best candidate: {result['best_candidate']}")

            # Log iteration stats
            if batch_rewards:
                mean_reward = np.mean(batch_rewards)
                self.reward_history.append(mean_reward)
                print(f"\n  Iteration {iteration+1} mean reward: {mean_reward:.2f}")

            # Save checkpoint
            if (iteration + 1) % self.config.save_frequency == 0:
                checkpoint_path = os.path.join(
                    self.config.output_dir,
                    f"checkpoint_iter_{iteration+1:04d}"
                )
                # TODO: Save model checkpoint
                print(f"  Checkpoint saved to {checkpoint_path}")

        print("\n" + "=" * 60)
        print("RLAIF TRAINING COMPLETE")
        print("=" * 60)


def main():
    config = RLAIFConfig()

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return

    trainer = RLAIFTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
