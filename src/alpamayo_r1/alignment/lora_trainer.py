# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoRA adapter trainer for fine-tuning AlpamayoR1 on preference data."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from alpamayo_r1.alignment.preference_collector import PreferenceCollector, PreferenceSample

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA training."""
    
    r: int = 8  # LoRA rank
    lora_alpha: int = 16
    target_modules: list[str] | None = None  # Auto-detect if None
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    # Training params
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    max_steps: int = -1  # Override epochs if > 0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Checkpointing
    save_steps: int = 50
    save_total_limit: int = 3


class LoRATrainer:
    """Trainer for LoRA adapters on AlpamayoR1 model."""
    
    def __init__(
        self,
        model: Any,  # AlpamayoR1 model
        output_dir: str | Path = "./lora_checkpoints",
        config: LoRAConfig | None = None,
    ):
        """Initialize the LoRA trainer.
        
        Args:
            model: The AlpamayoR1 model to fine-tune
            output_dir: Directory for saving checkpoints
            config: LoRA configuration
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or LoRAConfig()
        
        self._peft_model = None
        self._is_initialized = False
        self._training_step = 0
    
    def _check_peft_available(self) -> bool:
        """Check if PEFT library is available."""
        try:
            import peft
            return True
        except ImportError:
            return False
    
    def initialize_lora(self) -> None:
        """Initialize LoRA adapters on the model."""
        if not self._check_peft_available():
            raise ImportError(
                "PEFT library required for LoRA training. "
                "Install with: pip install peft"
            )
        
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Determine target modules
        target_modules = self.config.target_modules
        if target_modules is None:
            # Default for Qwen-based models
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        task_type = getattr(TaskType, self.config.task_type.upper(), TaskType.CAUSAL_LM)
        
        lora_config = LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            task_type=task_type,
        )
        
        # Get the VLM model for LoRA
        vlm = self.model.vlm if hasattr(self.model, "vlm") else self.model
        
        self._peft_model = get_peft_model(vlm, lora_config)
        self._peft_model.print_trainable_parameters()
        self._is_initialized = True
        
        logger.info("LoRA adapters initialized successfully")
    
    def _prepare_sft_batch(
        self,
        samples: list[PreferenceSample],
        processor: Any,
    ) -> dict[str, torch.Tensor]:
        """Prepare a batch for SFT training.
        
        Note: This is a simplified version. Full implementation would need
        to reconstruct the full input format with images and trajectory tokens.
        """
        # For now, we'll create a simple text-based training batch
        # Full implementation would need the processor and tokenizer
        
        texts = []
        for sample in samples:
            # Create training text
            text = f"<|cot_start|>{sample.chosen_reasoning}<|cot_end|>"
            texts.append(text)
        
        # This would need proper tokenization
        # For demonstration, return empty batch
        return {"input_ids": torch.tensor([]), "labels": torch.tensor([])}
    
    def train_step(
        self,
        samples: list[PreferenceSample],
        processor: Any = None,
    ) -> dict[str, float]:
        """Run a single training step on samples.
        
        Args:
            samples: List of preference samples to train on
            processor: The model processor for tokenization
            
        Returns:
            Dictionary with training metrics
        """
        if not self._is_initialized:
            self.initialize_lora()
        
        if not samples:
            return {"loss": 0.0, "samples": 0}
        
        # This is a simplified training step
        # Full implementation would need proper batch preparation
        
        self._training_step += 1
        
        # Log progress
        logger.info(f"Training step {self._training_step} on {len(samples)} samples")
        
        # For now, just return dummy metrics
        # Real implementation would compute actual loss
        return {
            "loss": 0.0,
            "step": self._training_step,
            "samples": len(samples),
        }
    
    def train_from_collector(
        self,
        collector: PreferenceCollector,
        processor: Any = None,
        mode: str = "sft",  # "sft" or "dpo"
    ) -> dict[str, Any]:
        """Train from collected preference data.
        
        Args:
            collector: The preference collector with samples
            processor: Model processor for tokenization
            mode: Training mode - "sft" for supervised, "dpo" for preference
            
        Returns:
            Training results
        """
        if not self._is_initialized:
            self.initialize_lora()
        
        if mode == "sft":
            samples = collector.get_sft_samples()
            logger.info(f"Training SFT on {len(samples)} samples")
        else:
            pairs = collector.get_dpo_pairs()
            samples = [chosen for chosen, _ in pairs]
            logger.info(f"Training DPO on {len(pairs)} preference pairs")
        
        if not samples:
            logger.warning("No samples to train on")
            return {"error": "No samples available"}
        
        # Simple batch training
        total_loss = 0.0
        num_batches = (len(samples) + self.config.batch_size - 1) // self.config.batch_size
        
        for i in range(0, len(samples), self.config.batch_size):
            batch = samples[i:i + self.config.batch_size]
            result = self.train_step(batch, processor)
            total_loss += result.get("loss", 0.0)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Save checkpoint
        checkpoint_path = self.save_checkpoint()
        
        return {
            "total_samples": len(samples),
            "num_batches": num_batches,
            "avg_loss": avg_loss,
            "checkpoint": str(checkpoint_path),
        }
    
    def save_checkpoint(self, name: str | None = None) -> Path:
        """Save current LoRA checkpoint.
        
        Args:
            name: Optional checkpoint name. If None, uses step number.
            
        Returns:
            Path to saved checkpoint
        """
        if name is None:
            name = f"checkpoint-{self._training_step}"
        
        checkpoint_path = self.output_dir / name
        
        if self._peft_model is not None:
            self._peft_model.save_pretrained(checkpoint_path)
            logger.info(f"Saved LoRA checkpoint to {checkpoint_path}")
        else:
            # Save a marker file if no PEFT model
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            (checkpoint_path / "training_state.txt").write_text(
                f"Step: {self._training_step}\n"
            )
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints keeping only save_total_limit."""
        checkpoints = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        
        for old_checkpoint in checkpoints[self.config.save_total_limit:]:
            import shutil
            shutil.rmtree(old_checkpoint)
            logger.info(f"Removed old checkpoint: {old_checkpoint}")
    
    def load_checkpoint(self, path: str | Path) -> None:
        """Load a LoRA checkpoint.
        
        Args:
            path: Path to the checkpoint directory
        """
        if not self._check_peft_available():
            raise ImportError("PEFT library required")
        
        from peft import PeftModel
        
        path = Path(path)
        vlm = self.model.vlm if hasattr(self.model, "vlm") else self.model
        
        self._peft_model = PeftModel.from_pretrained(vlm, path)
        self._is_initialized = True
        
        logger.info(f"Loaded LoRA checkpoint from {path}")
    
    @property
    def checkpoint_path(self) -> Path | None:
        """Get the latest checkpoint path."""
        checkpoints = list(self.output_dir.glob("checkpoint-*"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: p.stat().st_mtime)
