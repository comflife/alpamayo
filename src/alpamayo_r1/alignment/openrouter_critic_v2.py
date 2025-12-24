# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenRouter LLM for generating detailed off-road scene descriptions."""

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
class SceneDescription:
    """Result of scene description generation."""

    description: str
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "description": self.description,
            "raw_response": self.raw_response,
        }


class OffRoadDescriber:
    """LLM-based describer for generating detailed off-road scene descriptions."""

    def __init__(
        self,
        model: str = "google/gemini-3-flash-preview",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_image_size: int = 512,
        temperature: float = 0.7,
    ):
        """Initialize the OpenRouter describer.

        Args:
            model: The model to use for description generation
            api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY env var.
            base_url: OpenRouter API base URL.
            max_image_size: Maximum image dimension for resizing.
            temperature: Sampling temperature for generation (higher for more variety).
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
    
    def _build_description_prompt(self, guidelines: str = "") -> str:
        """Build prompt for detailed off-road scene description with reasoning."""
        base_prompt = """You are analyzing an off-road driving scene. Write ONE complete, detailed sentence describing the current terrain, hazards, and what should be considered for safe navigation.

Your sentence should include:
1. Terrain type (dirt/mud/grass/gravel/rocky/sandy)
2. Observable hazards or obstacles (puddles, deep mud, ruts, vegetation, rocks, trees, slopes, ditches)
3. Location and severity of hazards (left/right side, center, ahead, blocking the path, etc.)
4. Any relevant navigation considerations based on the terrain (without explicitly saying "turn" or "steer")

Write a detailed, natural sentence that flows well and provides complete context about the scene.

GOOD examples (like autonomous driving descriptions):
"Steer slightly right within the current path to maintain safe clearance from the deep muddy rut on the left side while keeping clear of the dense vegetation encroaching from the right edge."

"Navigate left around the large puddle blocking the center of the dirt path because the water depth is unclear and could hide deep mud or obstacles beneath."

"Keep to the right side of the grassy trail to avoid the rocky outcrop on the left and the overgrown bushes that are narrowing the passable area ahead."

"Maintain current trajectory on the dirt road while being aware of the steep ditch on the right side and scattered rocks on the left that could damage the vehicle."

BAD examples (too brief or incomplete):
"Muddy terrain with obstacles" (no detail, no context)
"There is a puddle" (incomplete - where? how big? what about it?)
"Avoid the rocks on the left" (too simple, missing terrain context)
"""

        if guidelines:
            base_prompt += f"""

# Off-Road Driving Guidelines

When analyzing the scene, consider these off-road driving rules:

{guidelines}

Use these guidelines to inform your description about what terrain conditions and hazards are important to identify.
"""

        base_prompt += """

Now write your ONE complete, detailed sentence about this off-road scene:"""

        return base_prompt
    
    def describe_scene(
        self,
        images: list[np.ndarray] | np.ndarray,
        guidelines: str = "",
    ) -> SceneDescription:
        """Generate detailed scene description from off-road images.

        Args:
            images: Single image or list of images (numpy arrays, HWC or CHW format)
            guidelines: Optional off-road driving guidelines to inform the description

        Returns:
            SceneDescription with detailed terrain analysis
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
                    "detail": "high"  # Use high detail for better descriptions
                }
            })

        # Add text prompt with guidelines
        prompt = self._build_description_prompt(guidelines=guidelines)
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
                max_tokens=1200,  # Enough for detailed reasoning with multiple hazards
                extra_body={"reasoning": {"enabled": True}},  # Enable deep reasoning
            )

            raw_response = response.choices[0].message.content
            description = raw_response.strip()

            # Remove any markdown formatting
            if description.startswith('"') and description.endswith('"'):
                description = description[1:-1]

            return SceneDescription(
                description=description,
                raw_response=raw_response,
            )

        except Exception as e:
            # Return error description
            return SceneDescription(
                description=f"[ERROR] Failed to generate description: {str(e)}",
                raw_response="",
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
