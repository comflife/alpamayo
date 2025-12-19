# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenRouter LLM Critic for evaluating model outputs against guidelines."""

import base64
import json
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Load .env file from alignment directory
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly

try:
    import openai
except ImportError:
    openai = None


@dataclass
class CritiqueResult:
    """Result of a critique evaluation."""
    
    violated: bool
    explanation: str
    corrected_reasoning: str | None = None
    corrected_trajectory: np.ndarray | None = None
    original_reasoning: str = ""
    original_trajectory: np.ndarray | None = None
    raw_response: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "violated": self.violated,
            "explanation": self.explanation,
            "corrected_reasoning": self.corrected_reasoning,
            "corrected_trajectory": self.corrected_trajectory.tolist() if self.corrected_trajectory is not None else None,
            "original_reasoning": self.original_reasoning,
            "original_trajectory": self.original_trajectory.tolist() if self.original_trajectory is not None else None,
        }


class OpenRouterCritic:
    """LLM-based critic using OpenRouter API for guideline compliance evaluation."""
    
    def __init__(
        self,
        model: str = "google/gemini-3-flash-preview",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_image_size: int = 512,
        temperature: float = 0.3,
    ):
        """Initialize the OpenRouter critic.
        
        Args:
            model: The model to use (default: openai/gpt-4o-mini)
            api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY env var.
            base_url: OpenRouter API base URL.
            max_image_size: Maximum image dimension for resizing.
            temperature: Sampling temperature for generation.
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
        self.max_image_size = max_image_size
        self.temperature = temperature
    
    def _image_to_base64(self, image: np.ndarray | Image.Image) -> str:
        """Convert image to base64 string."""
        if isinstance(image, np.ndarray):
            # Handle different formats
            if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                # CHW format -> HWC
                image = np.transpose(image, (1, 2, 0))
            if image.shape[-1] == 1:
                image = np.squeeze(image, axis=-1)
            image = Image.fromarray(image.astype(np.uint8))
        
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
    
    def _trajectory_to_text(self, trajectory: np.ndarray) -> str:
        """Convert trajectory array to text description."""
        if trajectory is None:
            return "No trajectory available"
        
        # trajectory shape: (T, 2) or (T, 3) for xy or xyz
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 2)
        
        # Summarize trajectory
        n_points = len(trajectory)
        start = trajectory[0]
        end = trajectory[-1]
        total_dist = np.sum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1))
        
        # Check for lateral movement (avoidance)
        lateral_movement = np.abs(trajectory[:, 1]).max()  # y-axis deviation
        
        return (
            f"Trajectory: {n_points} waypoints, "
            f"start=({start[0]:.2f}, {start[1]:.2f}), "
            f"end=({end[0]:.2f}, {end[1]:.2f}), "
            f"total_distance={total_dist:.2f}m, "
            f"max_lateral_deviation={lateral_movement:.2f}m"
        )
    
    def _build_prompt(
        self,
        reasoning: str,
        trajectory: np.ndarray,
        guidelines: str,
    ) -> str:
        """Build the evaluation prompt."""
        traj_text = self._trajectory_to_text(trajectory)
        
        # Include actual trajectory waypoints (first 10 and last 5 for brevity)
        waypoints_str = ""
        if trajectory is not None and len(trajectory) > 0:
            n = len(trajectory)
            if n <= 15:
                wp_subset = trajectory
            else:
                wp_subset = np.vstack([trajectory[:10], trajectory[-5:]])
            waypoints_str = "Waypoints (x, y in meters from ego):\n"
            for i, pt in enumerate(wp_subset[:10]):
                waypoints_str += f"  t+{i*0.1:.1f}s: ({pt[0]:.2f}, {pt[1]:.2f})\n"
            if n > 15:
                waypoints_str += "  ...\n"
                for j, pt in enumerate(wp_subset[10:]):
                    t = (n - 5 + j) * 0.1
                    waypoints_str += f"  t+{t:.1f}s: ({pt[0]:.2f}, {pt[1]:.2f})\n"
        
        return f"""You are an expert OFF-ROAD driving evaluator. 

## CORE RULE: ALWAYS steer to the CENTER of the WIDEST, MOST OPEN path!
- Find the widest open space in the image
- Trajectory must go through the CENTER of that open space
- Stay AWAY from edges, vegetation, trees, and obstacles

## ANALYZE THE IMAGE:
1. Where is the WIDEST, MOST OPEN area?
2. Is the trajectory going through the CENTER of that open space?
3. Or is it too close to vegetation/edges?

## ALPAMAYO OUTPUT:
Reasoning: \"{reasoning}\"
{traj_text}
{waypoints_str}

## DECISION:
- **VIOLATED** if trajectory is NOT centered in the open path
- **VIOLATED** if trajectory goes towards any vegetation or edge
- **OK** only if trajectory goes through the CENTER of the widest open area

## RESPONSE (JSON only):
{{
    "scene_description": "Open area is [where]. Vegetation is [where]. Current trajectory goes [where].",
    "violated": true or false,
    "explanation": "Trajectory [is/is not] centered. Should steer [direction] to center of open path.",
    "corrected_reasoning": "If violated: new reasoning. If OK: null",
    "trajectory_correction": {{
        "action": "left/right/straight/maintain",
        "lateral_offset_meters": 2.0,
        "description": "Steer to CENTER of open area"
    }}
}}

Be STRICT! The vehicle must stay in the CENTER of open paths."""
    
    def critique(
        self,
        images: list[np.ndarray] | np.ndarray,
        reasoning: str,
        trajectory: np.ndarray,
        guidelines: str,
    ) -> CritiqueResult:
        """Evaluate model output against guidelines.
        
        Args:
            images: Single image or list of images (numpy arrays, HWC or CHW format)
            reasoning: The model's chain-of-thought reasoning
            trajectory: The predicted trajectory (T, 2) or (T, 3)
            guidelines: The guidelines prompt string
            
        Returns:
            CritiqueResult with evaluation and corrections
        """
        # Normalize images to list
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
        elif isinstance(images, np.ndarray) and images.ndim == 4:
            images = [img for img in images]
        
        # Build message content
        content = []
        
        # Add images (limit to 4)
        for i, img in enumerate(images[:4]):
            b64 = self._image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low"
                }
            })
        
        # Add text prompt
        prompt = self._build_prompt(reasoning, trajectory, guidelines)
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
                max_tokens=1024,
            )
            
            raw_response = response.choices[0].message.content
            
            # Parse JSON response
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(raw_response)
            
            # Build corrected trajectory if violated
            corrected_traj = None
            if result.get("violated") and result.get("trajectory_correction"):
                correction = result["trajectory_correction"]
                action = correction.get("action", "maintain")
                offset = correction.get("lateral_offset_meters", 0)
                
                if trajectory is not None:
                    corrected_traj = trajectory.copy()
                    n_points = len(corrected_traj)
                    
                    # Create gradual offset: starts at 0, increases to full offset
                    # This keeps start point same but curves to new direction
                    t = np.linspace(0, 1, n_points)  # 0 to 1
                    gradual_factor = t ** 1.5  # Slightly curved ramp
                    
                    if action == "straight":
                        # Gradually reduce lateral component to 0
                        corrected_traj[:, 1] = corrected_traj[:, 1] * (1 - gradual_factor)
                    elif action == "left":
                        # Gradually add positive offset (left = +y)
                        offset = abs(offset) if offset != 0 else 2.0  # Default 2.0m
                        corrected_traj[:, 1] = corrected_traj[:, 1] + gradual_factor * offset
                    elif action == "right":
                        # Gradually add negative offset (right = -y)
                        offset = abs(offset) if offset != 0 else 2.0  # Default 2.0m  
                        corrected_traj[:, 1] = corrected_traj[:, 1] - gradual_factor * offset
            
            return CritiqueResult(
                violated=result.get("violated", False),
                explanation=result.get("explanation", ""),
                corrected_reasoning=result.get("corrected_reasoning"),
                corrected_trajectory=corrected_traj,
                original_reasoning=reasoning,
                original_trajectory=trajectory,
                raw_response=raw_response,
            )
            
        except json.JSONDecodeError as e:
            return CritiqueResult(
                violated=False,
                explanation=f"Failed to parse response: {e}",
                original_reasoning=reasoning,
                original_trajectory=trajectory,
                raw_response=raw_response if 'raw_response' in locals() else "",
            )
        except Exception as e:
            return CritiqueResult(
                violated=False,
                explanation=f"API error: {e}",
                original_reasoning=reasoning,
                original_trajectory=trajectory,
            )
    
    def test_connection(self) -> dict[str, Any]:
        """Test the API connection."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": "Say 'OK' if you can read this."
                }],
                max_tokens=10,
            )
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "model": self.model,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": self.model,
            }
