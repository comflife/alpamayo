#!/usr/bin/env python3
"""
RLAIF Evaluator: Uses OpenRouter LLM to score trajectory + reasoning candidates.

Unlike OpenRouterCritic which returns violated/corrected,
this evaluator returns numerical scores for RL training.
"""

import os
import json
import re
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from typing import List, Dict, Tuple
from dataclasses import dataclass

try:
    import openai
except ImportError:
    openai = None


@dataclass
class RLAIFScores:
    """Scores for a single (reasoning, trajectory) candidate."""
    trajectory_quality: float  # 0-10
    reasoning_quality: float   # 0-10
    consistency: float         # 0-10
    explanation: str
    overall_reward: float      # Average of the three scores


class RLAIFEvaluator:
    """Evaluates multiple candidates and returns scores for RL training."""

    def __init__(
        self,
        model: str = "google/gemini-3-flash-preview",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.3,
        max_image_size: int = 512,
    ):
        """Initialize the RLAIF evaluator.

        Args:
            model: The model to use for evaluation
            api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY env var.
            base_url: OpenRouter API base URL
            temperature: Sampling temperature
            max_image_size: Maximum image dimension for resizing
        """
        if openai is None:
            raise ImportError("openai package required. Install with: pip install openai")

        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=self.api_key,
        )
        self.temperature = temperature
        self.max_image_size = max_image_size

    def _image_to_base64(self, image: np.ndarray | Image.Image) -> str:
        """Convert image to base64 string."""
        if isinstance(image, np.ndarray):
            # Handle different formats
            if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                # CHW format -> HWC
                image = np.transpose(image, (1, 2, 0))
            if image.shape[-1] == 1:
                image = np.squeeze(image, axis=-1)

            # Ensure uint8
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            image = Image.fromarray(image)

        # Resize if needed
        w, h = image.size
        if max(w, h) > self.max_image_size:
            scale = self.max_image_size / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _trajectory_summary(self, trajectory: np.ndarray) -> Dict[str, any]:
        """Get trajectory statistics."""
        if trajectory is None or len(trajectory) == 0:
            return {"direction": "unknown", "max_lateral": 0.0}

        # trajectory shape: (T, 2) for xy
        lateral_offset = trajectory[-1, 1]  # Final y position
        max_lateral = np.abs(trajectory[:, 1]).max()

        if abs(lateral_offset) < 0.5:
            direction = "straight"
        elif lateral_offset > 0.5:
            direction = "left"
        else:
            direction = "right"

        return {
            "direction": direction,
            "max_lateral": max_lateral,
            "final_lateral": lateral_offset,
        }

    def _build_evaluation_prompt(
        self,
        trajectories: List[np.ndarray],
        reasonings: List[str],
        guidelines: str,
    ) -> str:
        """Build prompt for evaluating multiple candidates."""

        prompt = f"""# ROLE
You are an expert OFF-ROAD AUTONOMOUS DRIVING EVALUATOR.

# GUIDELINES
{guidelines}

# YOUR TASK
You are shown a camera image with {len(trajectories)} different trajectory candidates.
Each candidate has:
- A trajectory path (waypoints in meters from ego vehicle)
- A reasoning explanation (text describing the decision)

For each candidate, evaluate:
1. **Trajectory Quality (0-10)**: How well does the trajectory follow the guidelines?
   - Does it follow visible tire tracks/paths?
   - Does it avoid puddles, deep mud, ruts, and obstacles?
   - Is it smooth and consistent?
   - Does it maintain safe clearance from edges and vegetation?

2. **Reasoning Quality (0-10)**: How accurately does the reasoning describe the scene?
   - Does it correctly identify terrain features (puddles, paths, obstacles)?
   - Does it explain the trajectory choice clearly?
   - Is it specific and actionable?

3. **Consistency (0-10)**: Do the trajectory and reasoning match?
   - If reasoning says "turn right", does the trajectory turn right?
   - If reasoning mentions avoiding a puddle, does the trajectory actually avoid it?

# CANDIDATES

"""

        for i, (traj, reasoning) in enumerate(zip(trajectories, reasonings)):
            summary = self._trajectory_summary(traj)
            prompt += f"""
## Candidate {i+1}
- **Reasoning**: {reasoning}
- **Trajectory Stats**: Direction={summary['direction']}, Max lateral offset={summary['max_lateral']:.2f}m, Final lateral={summary['final_lateral']:.2f}m

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
- 0-3: Poor/dangerous
- 4-6: Acceptable but needs improvement
- 7-8: Good
- 9-10: Excellent

Output ONLY the JSON, no other text.
"""
        return prompt

    def evaluate_candidates(
        self,
        image: np.ndarray,
        trajectories: List[np.ndarray],
        reasonings: List[str],
        guidelines: str,
    ) -> List[RLAIFScores]:
        """Evaluate multiple candidates and return scores.

        Args:
            image: Camera image (HWC or CHW numpy array)
            trajectories: List of trajectory arrays, each (T, 2)
            reasonings: List of reasoning strings
            guidelines: Guidelines prompt string

        Returns:
            List of RLAIFScores, one per candidate
        """
        if len(trajectories) != len(reasonings):
            raise ValueError("Number of trajectories must match number of reasonings")

        # Build message content
        content = []

        # Add image
        b64 = self._image_to_base64(image)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "low"
            }
        })

        # Add prompt
        prompt = self._build_evaluation_prompt(trajectories, reasonings, guidelines)
        content.append({
            "type": "text",
            "text": prompt
        })

        # Call API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": content
                }],
                temperature=self.temperature,
                max_tokens=4096,
                extra_body={"reasoning": {"enabled": True}}  # Enable deep reasoning
            )

            raw_response = response.choices[0].message.content

            # Parse JSON
            cleaned_response = raw_response
            if "```json" in cleaned_response:
                cleaned_response = cleaned_response.replace("```json", "").replace("```", "")
            elif "```" in cleaned_response:
                cleaned_response = cleaned_response.replace("```", "")
            cleaned_response = cleaned_response.strip()

            # Extract JSON
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(cleaned_response)

            # Convert to RLAIFScores
            scores_list = []
            for eval_item in result.get("evaluations", []):
                tq = float(eval_item.get("trajectory_quality", 5.0))
                rq = float(eval_item.get("reasoning_quality", 5.0))
                cons = float(eval_item.get("consistency", 5.0))

                scores = RLAIFScores(
                    trajectory_quality=tq,
                    reasoning_quality=rq,
                    consistency=cons,
                    explanation=eval_item.get("explanation", ""),
                    overall_reward=(tq + rq + cons) / 3.0,
                )
                scores_list.append(scores)

            # Ensure we have scores for all candidates
            while len(scores_list) < len(trajectories):
                # Default scores if LLM didn't return enough
                scores_list.append(RLAIFScores(
                    trajectory_quality=5.0,
                    reasoning_quality=5.0,
                    consistency=5.0,
                    explanation="LLM did not provide evaluation",
                    overall_reward=5.0,
                ))

            return scores_list

        except Exception as e:
            # Return default scores on error
            print(f"RLAIF evaluation error: {e}")
            return [
                RLAIFScores(
                    trajectory_quality=5.0,
                    reasoning_quality=5.0,
                    consistency=5.0,
                    explanation=f"Evaluation failed: {e}",
                    overall_reward=5.0,
                )
                for _ in range(len(trajectories))
            ]
