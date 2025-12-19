# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preference data collector for DPO/SFT training from critique results."""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from alpamayo_r1.alignment.openrouter_critic import CritiqueResult


@dataclass
class PreferenceSample:
    """A single preference sample for training."""
    
    # Images (stored as file paths to avoid memory issues)
    image_paths: list[str] = field(default_factory=list)
    
    # Original model output
    original_reasoning: str = ""
    original_trajectory: list[list[float]] | None = None
    
    # Corrected output (if violated)
    chosen_reasoning: str = ""
    chosen_trajectory: list[list[float]] | None = None
    
    # Metadata
    violated: bool = False
    explanation: str = ""
    timestamp: str = ""
    clip_id: str = ""
    t0_us: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "PreferenceSample":
        return cls(**data)


class PreferenceCollector:
    """Collects preference data from critique results for training."""
    
    def __init__(
        self,
        save_dir: str | Path = "./preference_data",
        auto_save_interval: int = 10,
    ):
        """Initialize the preference collector.
        
        Args:
            save_dir: Directory to save preference data
            auto_save_interval: Auto-save after this many samples
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.save_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        self.auto_save_interval = auto_save_interval
        self.samples: list[PreferenceSample] = []
        self._sample_count = 0
        
        # Load existing samples if any
        self._load_existing()
    
    def _load_existing(self) -> None:
        """Load existing samples from save directory."""
        data_file = self.save_dir / "preferences.jsonl"
        if data_file.exists():
            with open(data_file, "r") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(PreferenceSample.from_dict(json.loads(line)))
            self._sample_count = len(self.samples)
    
    def _save_images(
        self,
        images: list[np.ndarray],
        prefix: str,
    ) -> list[str]:
        """Save images to disk and return paths."""
        from PIL import Image
        
        paths = []
        for i, img in enumerate(images):
            # Convert if needed
            if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 1:
                img = np.squeeze(img, axis=-1)
            
            filename = f"{prefix}_img{i}.jpg"
            path = self.images_dir / filename
            Image.fromarray(img.astype(np.uint8)).save(path, quality=85)
            paths.append(str(path))
        
        return paths
    
    def add(
        self,
        critique_result: CritiqueResult,
        images: list[np.ndarray] | np.ndarray | None = None,
        clip_id: str = "",
        t0_us: int = 0,
    ) -> PreferenceSample:
        """Add a sample from a critique result.
        
        Args:
            critique_result: The critique result from OpenRouterCritic
            images: Optional images to save (numpy arrays)
            clip_id: Dataset clip identifier
            t0_us: Timestamp in microseconds
            
        Returns:
            The created PreferenceSample
        """
        self._sample_count += 1
        timestamp = datetime.now().isoformat()
        prefix = f"sample_{self._sample_count}_{timestamp.replace(':', '-')}"
        
        # Save images
        image_paths = []
        if images is not None:
            if isinstance(images, np.ndarray) and images.ndim == 3:
                images = [images]
            elif isinstance(images, np.ndarray) and images.ndim == 4:
                images = [img for img in images]
            image_paths = self._save_images(images, prefix)
        
        # Convert trajectories
        orig_traj = None
        if critique_result.original_trajectory is not None:
            orig_traj = critique_result.original_trajectory.tolist()
        
        chosen_traj = None
        chosen_reasoning = critique_result.original_reasoning
        
        if critique_result.violated:
            # If violated, use corrected versions
            if critique_result.corrected_trajectory is not None:
                chosen_traj = critique_result.corrected_trajectory.tolist()
            else:
                chosen_traj = orig_traj  # Fallback
            
            if critique_result.corrected_reasoning:
                chosen_reasoning = critique_result.corrected_reasoning
        else:
            # If not violated, original is the chosen output
            chosen_traj = orig_traj
            chosen_reasoning = critique_result.original_reasoning
        
        sample = PreferenceSample(
            image_paths=image_paths,
            original_reasoning=critique_result.original_reasoning,
            original_trajectory=orig_traj,
            chosen_reasoning=chosen_reasoning,
            chosen_trajectory=chosen_traj,
            violated=critique_result.violated,
            explanation=critique_result.explanation,
            timestamp=timestamp,
            clip_id=clip_id,
            t0_us=t0_us,
        )
        
        self.samples.append(sample)
        
        # Auto-save
        if len(self.samples) % self.auto_save_interval == 0:
            self.save()
        
        return sample
    
    def save(self) -> None:
        """Save all samples to JSONL file."""
        data_file = self.save_dir / "preferences.jsonl"
        with open(data_file, "w") as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict()) + "\n")
    
    def get_dpo_pairs(self) -> list[tuple[PreferenceSample, PreferenceSample]]:
        """Get DPO training pairs (chosen, rejected) from violated samples."""
        pairs = []
        for sample in self.samples:
            if sample.violated:
                # Create a "rejected" version with original outputs
                rejected = PreferenceSample(
                    image_paths=sample.image_paths,
                    original_reasoning=sample.original_reasoning,
                    original_trajectory=sample.original_trajectory,
                    chosen_reasoning=sample.original_reasoning,  # Use original as "chosen" for rejected
                    chosen_trajectory=sample.original_trajectory,
                    violated=True,
                    explanation=sample.explanation,
                    timestamp=sample.timestamp,
                    clip_id=sample.clip_id,
                    t0_us=sample.t0_us,
                )
                
                # Current sample has corrected versions as "chosen"
                pairs.append((sample, rejected))
        
        return pairs
    
    def get_sft_samples(self) -> list[PreferenceSample]:
        """Get all samples for SFT training (chosen outputs only)."""
        return self.samples
    
    @property
    def size(self) -> int:
        """Number of samples collected."""
        return len(self.samples)
    
    @property
    def violated_count(self) -> int:
        """Number of violated samples."""
        return sum(1 for s in self.samples if s.violated)
    
    def stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        return {
            "total_samples": self.size,
            "violated_samples": self.violated_count,
            "compliant_samples": self.size - self.violated_count,
            "violation_rate": self.violated_count / self.size if self.size > 0 else 0,
        }
    
    def clear(self) -> None:
        """Clear all collected samples."""
        self.samples = []
        self._sample_count = 0
