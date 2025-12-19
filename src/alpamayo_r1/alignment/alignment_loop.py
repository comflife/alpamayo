# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main alignment loop orchestrating the self-improvement pipeline."""

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from alpamayo_r1.alignment.guidelines import Guidelines, load_guidelines
from alpamayo_r1.alignment.openrouter_critic import OpenRouterCritic, CritiqueResult
from alpamayo_r1.alignment.preference_collector import PreferenceCollector
from alpamayo_r1.alignment.lora_trainer import LoRATrainer, LoRAConfig

logger = logging.getLogger(__name__)


class AlignmentLoop:
    """Main orchestrator for the guideline alignment self-improvement loop."""
    
    def __init__(
        self,
        model: Any,  # AlpamayoR1 model
        guidelines_path: str | Path | None = None,
        openrouter_model: str = "openai/gpt-4o-mini",
        output_dir: str | Path = "./alignment_output",
        update_threshold: int = 10,  # Update LoRA every N samples
        lora_config: LoRAConfig | None = None,
    ):
        """Initialize the alignment loop.
        
        Args:
            model: The AlpamayoR1 model
            guidelines_path: Path to guidelines YAML file
            openrouter_model: OpenRouter model for critique
            output_dir: Base directory for outputs
            update_threshold: Number of samples before LoRA update
            lora_config: LoRA training configuration
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.guidelines = load_guidelines(guidelines_path)
        self.guidelines_prompt = self.guidelines.to_prompt()
        
        self.critic = OpenRouterCritic(model=openrouter_model)
        
        self.collector = PreferenceCollector(
            save_dir=self.output_dir / "preference_data",
        )
        
        self.trainer = LoRATrainer(
            model=model,
            output_dir=self.output_dir / "lora_checkpoints",
            config=lora_config,
        )
        
        self.update_threshold = update_threshold
        self._samples_since_update = 0
        self._total_processed = 0
        
        logger.info(f"AlignmentLoop initialized with {len(self.guidelines)} guideline rules")
        logger.info(f"Update threshold: {update_threshold} samples")
    
    def process_sample(
        self,
        images: np.ndarray | list[np.ndarray],
        reasoning: str,
        trajectory: np.ndarray,
        clip_id: str = "",
        t0_us: int = 0,
    ) -> CritiqueResult:
        """Process a single sample through the alignment pipeline.
        
        Args:
            images: Input images (N, C, H, W) or list of (C, H, W)
            reasoning: Model's chain-of-thought reasoning
            trajectory: Predicted trajectory (T, 2) or (T, 3)
            clip_id: Dataset clip identifier
            t0_us: Timestamp in microseconds
            
        Returns:
            CritiqueResult from the LLM critic
        """
        self._total_processed += 1
        
        # Critique the sample
        critique = self.critic.critique(
            images=images,
            reasoning=reasoning,
            trajectory=trajectory,
            guidelines=self.guidelines_prompt,
        )
        
        # Log result
        status = "VIOLATED" if critique.violated else "OK"
        logger.info(f"Sample {self._total_processed}: {status} - {critique.explanation[:100]}...")
        
        # Collect preference data
        self.collector.add(
            critique_result=critique,
            images=images if isinstance(images, list) else [images] if images.ndim == 3 else list(images),
            clip_id=clip_id,
            t0_us=t0_us,
        )
        
        self._samples_since_update += 1
        
        # Check if we should update LoRA
        if self._samples_since_update >= self.update_threshold:
            self._run_update()
        
        return critique
    
    def _run_update(self) -> dict[str, Any]:
        """Run a LoRA training update."""
        logger.info(f"Running LoRA update after {self._samples_since_update} samples...")
        
        result = self.trainer.train_from_collector(
            collector=self.collector,
            mode="sft",  # Use SFT for simplicity
        )
        
        self._samples_since_update = 0
        logger.info(f"LoRA update complete: {result}")
        
        return result
    
    def run_on_dataset(
        self,
        data_iterator: Iterator[dict[str, Any]],
        max_samples: int | None = None,
    ) -> dict[str, Any]:
        """Run alignment loop on a dataset iterator.
        
        Args:
            data_iterator: Iterator yielding dicts with keys:
                - images: np.ndarray
                - reasoning: str
                - trajectory: np.ndarray
                - clip_id: str (optional)
                - t0_us: int (optional)
            max_samples: Maximum number of samples to process
            
        Returns:
            Summary statistics
        """
        processed = 0
        
        for data in data_iterator:
            if max_samples and processed >= max_samples:
                break
            
            self.process_sample(
                images=data["images"],
                reasoning=data["reasoning"],
                trajectory=data["trajectory"],
                clip_id=data.get("clip_id", ""),
                t0_us=data.get("t0_us", 0),
            )
            processed += 1
        
        # Final save
        self.collector.save()
        
        return {
            "total_processed": processed,
            **self.collector.stats(),
            "lora_checkpoints": str(self.trainer.checkpoint_path),
        }
    
    def run_single_inference_and_align(
        self,
        data: dict[str, Any],
        device: str = "cuda",
    ) -> dict[str, Any]:
        """Run inference on data and then process through alignment.
        
        This method runs the full pipeline:
        1. Model inference to get reasoning and trajectory
        2. Critique with OpenRouter
        3. Collect preference data
        4. Periodically update LoRA
        
        Args:
            data: Input data with keys from load_physical_aiavdataset
            device: Device for inference
            
        Returns:
            Dictionary with inference results and critique
        """
        from alpamayo_r1 import helper
        
        # Prepare inputs
        processor = helper.get_processor(self.model.tokenizer)
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
        model_inputs = helper.to_device(model_inputs, device)
        
        # Run inference
        with torch.autocast(device, dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=True,
            )
        
        # Extract results
        reasoning = extra["cot"][0] if extra.get("cot") else ""
        trajectory = pred_xyz.cpu().numpy()[0, 0, 0]  # (T, 3)
        
        # Convert images for critique (take first 4)
        images_np = images[:4].permute(0, 2, 3, 1).cpu().numpy()  # (4, H, W, C)
        
        # Process through alignment
        critique = self.process_sample(
            images=list(images_np),
            reasoning=reasoning,
            trajectory=trajectory,
            clip_id=data.get("clip_id", ""),
            t0_us=data.get("t0_us", 0),
        )
        
        return {
            "pred_xyz": pred_xyz,
            "pred_rot": pred_rot,
            "reasoning": reasoning,
            "critique": critique.to_dict(),
            "violated": critique.violated,
            "corrected_reasoning": critique.corrected_reasoning,
            "corrected_trajectory": critique.corrected_trajectory,
        }
    
    def stats(self) -> dict[str, Any]:
        """Get current statistics."""
        return {
            "total_processed": self._total_processed,
            "samples_since_update": self._samples_since_update,
            "update_threshold": self.update_threshold,
            **self.collector.stats(),
        }


def run_alignment_demo(
    clip_id: str,
    t0_us: int = 5_100_000,
    guidelines_path: str | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run a single demo of the alignment loop.
    
    Args:
        clip_id: Dataset clip ID
        t0_us: Timestamp in microseconds
        guidelines_path: Path to guidelines YAML (uses default if None)
        device: Device for inference
        
    Returns:
        Alignment results
    """
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    
    print(f"Loading dataset for clip_id: {clip_id}...")
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    data["clip_id"] = clip_id
    data["t0_us"] = t0_us
    
    print("Loading model...")
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to(device)
    
    print("Initializing alignment loop...")
    loop = AlignmentLoop(
        model=model,
        guidelines_path=guidelines_path,
        openrouter_model="openai/gpt-4o-mini",
        update_threshold=5,  # Small threshold for demo
    )
    
    print("Running inference and alignment...")
    result = loop.run_single_inference_and_align(data, device=device)
    
    print("\n" + "="*50)
    print("ALIGNMENT RESULT")
    print("="*50)
    print(f"Violated: {result['violated']}")
    print(f"\nOriginal Reasoning:\n{result['reasoning'][:500]}...")
    print(f"\nCritique:\n{result['critique']['explanation']}")
    if result['violated'] and result['corrected_reasoning']:
        print(f"\nCorrected Reasoning:\n{result['corrected_reasoning'][:500]}...")
    
    print(f"\nStats: {loop.stats()}")
    
    return result


if __name__ == "__main__":
    # Example usage
    import sys
    
    clip_id = sys.argv[1] if len(sys.argv) > 1 else "030c760c-ae38-49aa-9ad8-f5650a545d26"
    run_alignment_demo(clip_id)
