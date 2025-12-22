"""
Full Pipeline Fine-tuning for Alpamayo-R1.
Trains VLM + Expert + Diffusion with combined language + trajectory loss.

Usage:
cd /home/byounggun/alpamayo/src
torchrun --nproc_per_node=4 -m alpamayo_r1.alignment.finetune_full \
    --data_path /home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset/finetune_data.jsonl \
    --output_dir /home/byounggun/alpamayo/outputs/alpamayo_full_finetuned \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --save_steps 100 \
    --bf16 True
"""

import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import numpy as np
from PIL import Image

import transformers
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import einops

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# Fix for RTX 4000 series DDP issues
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="nvidia/Alpamayo-R1-10B")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the jsonl dataset"})


@dataclass
class FullTrainingArguments(transformers.TrainingArguments):
    """Training arguments for full pipeline fine-tuning."""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    max_length: int = field(default=1024)
    
    # LoRA config - increased for better capacity
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    
    # Loss weighting
    traj_loss_weight: float = field(default=1.0, metadata={"help": "Weight for trajectory loss"})
    
    # Memory optimization
    gradient_checkpointing: bool = False  # Disable for device_map=auto
    
    # Disable DataParallel when using device_map=auto
    dataloader_pin_memory: bool = False


class AlpamayoFullModel(nn.Module):
    """
    Wrapper for full Alpamayo model training.
    Handles both language loss (VLM) and trajectory loss (Expert + Diffusion).
    """
    
    def __init__(self, base_model: AlpamayoR1, traj_loss_weight: float = 1.0):
        super().__init__()
        self.base_model = base_model
        self.traj_loss_weight = traj_loss_weight
        
        # Components
        self.vlm = base_model.vlm
        self.expert = base_model.expert
        self.diffusion = base_model.diffusion
        self.action_space = base_model.action_space
        self.action_in_proj = base_model.action_in_proj
        self.action_out_proj = base_model.action_out_proj
        self.config = base_model.config
        
        # Freeze VLM, only train Expert + Diffusion-related parts
        # Actually: VLM already has LoRA applied, so LoRA params are trainable
        
    @property
    def tokenizer(self):
        return self.base_model.tokenizer
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for VLM."""
        if hasattr(self.vlm, 'gradient_checkpointing_enable'):
            self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        elif hasattr(self.vlm, 'base_model') and hasattr(self.vlm.base_model, 'gradient_checkpointing_enable'):
            self.vlm.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for VLM."""
        if hasattr(self.vlm, 'gradient_checkpointing_disable'):
            self.vlm.gradient_checkpointing_disable()
        elif hasattr(self.vlm, 'base_model') and hasattr(self.vlm.base_model, 'gradient_checkpointing_disable'):
            self.vlm.base_model.gradient_checkpointing_disable()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        gt_trajectory: Optional[torch.Tensor] = None,
        ego_history_xyz: Optional[torch.Tensor] = None,
        ego_history_rot: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass with COMBINED language + trajectory loss.
        
        Args:
            input_ids: Token ids
            attention_mask: Attention mask
            labels: Labels for language modeling
            pixel_values: Image features
            image_grid_thw: Image grid info for Qwen3-VL
            gt_trajectory: Ground truth trajectory (B, T, 2 or 3)
            ego_history_xyz: Ego history positions
            ego_history_rot: Ego history rotations
        """
        device = input_ids.device
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        # =================================================================
        # 1. Language Loss (VLM forward)
        # =================================================================
        vlm_kwargs = {}
        if pixel_values is not None:
            vlm_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            vlm_kwargs["image_grid_thw"] = image_grid_thw
            
        vlm_outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **vlm_kwargs,
        )
        
        language_loss = vlm_outputs.loss
        if language_loss is not None:
            total_loss = total_loss + language_loss
        
        # =================================================================
        # 2. Trajectory Loss (Flow Matching)
        # =================================================================
        if gt_trajectory is not None and ego_history_xyz is not None:
            traj_loss = self._compute_trajectory_loss(
                gt_trajectory=gt_trajectory,
                ego_history_xyz=ego_history_xyz,
                ego_history_rot=ego_history_rot,
            )
            total_loss = total_loss + self.traj_loss_weight * traj_loss
        
        return {"loss": total_loss}
    
    def _compute_trajectory_loss(
        self,
        gt_trajectory: torch.Tensor,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
        vlm_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute trajectory loss using flow matching.
        
        Flow matching loss: MSE(pred_v, target_v)
        where target_v = x_1 - x_0 (straight line path)
        x_0 ~ N(0, 1), x_1 = gt_action
        """
        B = gt_trajectory.shape[0]
        
        # Get device and dtype from expert module
        try:
            param = next(self.expert.parameters())
            device = param.device
            dtype = param.dtype  # Should be bfloat16
        except StopIteration:
            device = gt_trajectory.device
            dtype = torch.bfloat16
        
        # Move input tensors to correct device AND dtype
        gt_trajectory = gt_trajectory.to(device=device, dtype=dtype)
        ego_history_xyz = ego_history_xyz.to(device=device, dtype=dtype)
        if ego_history_rot is not None:
            ego_history_rot = ego_history_rot.to(device=device, dtype=dtype)
        
        # Ensure trajectory has z=0 if only xy provided
        if gt_trajectory.shape[-1] == 2:
            gt_trajectory = F.pad(gt_trajectory, (0, 1), value=0)  # Add z=0
        
        # We need rotation matrices for action space conversion
        # If ego_history_rot is None, use identity
        if ego_history_rot is None or ego_history_rot.numel() == 0:
            # Create identity rotation for each trajectory point
            T = gt_trajectory.shape[1]
            ego_history_rot = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            ego_history_rot = ego_history_rot.expand(B, 1, T, 3, 3)
        
        # Ensure ego_history has correct shape
        # Expected: (B, 1, T, 3) for xyz, (B, 1, T, 3, 3) for rot
        if ego_history_xyz.dim() == 3:
            ego_history_xyz = ego_history_xyz.unsqueeze(1)  # Add sample dim
        if ego_history_rot.dim() == 4:
            ego_history_rot = ego_history_rot.unsqueeze(1)
            
        # Convert trajectory to action space
        # action_space expects: (B, T, 3) for xyz, (B, T, 3, 3) for rot
        # We need to create future_rot from trajectory direction or use identity
        T_future = gt_trajectory.shape[1]
        gt_future_rot = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, T_future, 3, 3).clone()
        
        # Need to reshape for action_space
        hist_xyz = ego_history_xyz[:, 0]  # (B, T_hist, 3)
        hist_rot = ego_history_rot[:, 0]  # (B, T_hist, 3, 3)
        
        try:
            # IMPORTANT: cholesky operation requires float32, not bfloat16
            # Convert to float32 for action space conversion
            gt_action = self.action_space.traj_to_action(
                traj_history_xyz=hist_xyz.float(),
                traj_history_rot=hist_rot.float(),
                traj_future_xyz=gt_trajectory.float(),
                traj_future_rot=gt_future_rot.float(),
            )  # (B, *action_dims) in float32
            # Convert back to bfloat16
            gt_action = gt_action.to(dtype)
        except Exception as e:
            logger.warning(f"Failed to convert trajectory to action: {e}")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # ==== Flow Matching Training ====
        # Sample random timestep t ~ U(0, 1)
        t = torch.rand(B, device=device, dtype=dtype).view(B, 1, 1)  # (B, 1, 1)
        
        # Sample noise x_0 ~ N(0, 1)
        x_0 = torch.randn(gt_action.shape, device=device, dtype=dtype)  # (B, *action_dims)
        
        # Interpolate: x_t = (1-t) * x_0 + t * x_1
        x_t = (1 - t) * x_0 + t * gt_action
        
        # Target vector field: v = x_1 - x_0
        target_v = gt_action - x_0
        
        # We need hidden states from VLM to condition the expert
        # For simplified training, we use a dummy conditioning
        # In full implementation, we'd use vlm_hidden_states or run VLM generate
        
        # Predict vector field using action_in_proj -> expert -> action_out_proj
        n_tokens = self.action_space.get_action_space_dims()[0]
        
        # Project noisy action to embeddings
        action_embeds = self.action_in_proj(x_t, t.squeeze(-1).squeeze(-1))
        if action_embeds.dim() == 2:
            action_embeds = action_embeds.view(B, n_tokens, -1)
        
        # Run expert (simplified: no KV cache conditioning for training)
        # This is a simplification - ideally we'd cache VLM outputs
        expert_out = self.expert(
            inputs_embeds=action_embeds,
            use_cache=False,
        )
        
        last_hidden = expert_out.last_hidden_state  # (B, n_tokens, hidden)
        pred_v = self.action_out_proj(last_hidden)  # (B, n_tokens, action_dim)
        pred_v = pred_v.view(B, *self.action_space.get_action_space_dims())
        
        # MSE loss
        loss = F.mse_loss(pred_v, target_v)
        
        return loss


class FullPipelineDataset(torch.utils.data.Dataset):
    """Dataset that includes trajectory supervision."""
    
    def __init__(self, data_path: str, processor, max_length: int = 2048):
        super().__init__()
        self.processor = processor
        self.max_length = max_length
        
        logging.warning("Loading data...")
        with open(data_path, "r") as f:
            self.list_data_dict = [json.loads(line) for line in f]
        
        logging.warning(f"Loaded {len(self.list_data_dict)} examples.")
    
    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[i]
        
        # Load images
        frames = []
        for fp in sample['frame_paths']:
            img = Image.open(fp).convert("RGB")
            img_np = np.array(img)
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            frames.append(img_t)
        
        frames_tensor = torch.stack(frames, dim=0)
        
        # Create message
        messages = helper.create_message(frames_tensor)
        
        # Add assistant response
        reasoning = sample['reasoning']
        messages.append({
            "role": "assistant", 
            "content": [{"type": "text", "text": reasoning + "<|traj_future_start|>"}]
        })
        
        # Tokenize full conversation
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        
        # Compute prompt length for masking
        prompt_messages = messages[:-1]
        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]
        
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        # Include pixel values
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            if pv.dim() > 3:
                pv = pv.squeeze(0)
            output["pixel_values"] = pv
        if "image_grid_thw" in inputs:
            igt = inputs["image_grid_thw"]
            if igt.dim() == 3:
                igt = igt.squeeze(0)
            output["image_grid_thw"] = igt
        
        # Include trajectory
        if 'trajectory' in sample and sample['trajectory'] is not None:
            traj = np.array(sample['trajectory'])
            output["gt_trajectory"] = torch.from_numpy(traj).float()
        
        # Create ego history (simplified straight line)
        num_history = 16
        positions = np.zeros((num_history, 3))
        positions[:, 0] = np.linspace(-7.5, 0, num_history)  # Forward motion history
        output["ego_history_xyz"] = torch.from_numpy(positions).float()
        
        rot = np.eye(3)[None].repeat(num_history, axis=0)
        output["ego_history_rot"] = torch.from_numpy(rot).float()
        
        return output


def collate_fn(instances, tokenizer):
    """Custom collator for full pipeline training."""
    batch = {}
    keys = instances[0].keys()
    
    for k in keys:
        vals = [i[k] for i in instances if k in i]
        if len(vals) == 0:
            continue
            
        if k in ["input_ids", "labels"]:
            batch[k] = torch.nn.utils.rnn.pad_sequence(
                vals, batch_first=True, padding_value=tokenizer.pad_token_id
            )
        elif k == "attention_mask":
            batch[k] = torch.nn.utils.rnn.pad_sequence(
                vals, batch_first=True, padding_value=0
            )
        elif k in ["pixel_values", "image_grid_thw"]:
            batch[k] = torch.cat(vals, dim=0)
        elif k == "gt_trajectory":
            # Pad trajectories to same length
            max_len = max(v.shape[0] for v in vals)
            padded = []
            for v in vals:
                if v.shape[0] < max_len:
                    pad_size = max_len - v.shape[0]
                    v = F.pad(v, (0, 0, 0, pad_size))
                padded.append(v)
            batch[k] = torch.stack(padded)
        elif k in ["ego_history_xyz", "ego_history_rot"]:
            batch[k] = torch.stack(vals)
        else:
            try:
                batch[k] = torch.stack(vals)
            except:
                batch[k] = vals
    
    return batch


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, FullTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print(f"Loading Alpamayo model from {model_args.model_name_or_path}...")
    print("Using device_map='auto' to shard model across available GPUs...")
    base_model = AlpamayoR1.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.bfloat16,
        device_map="auto",  # Automatically shard across GPUs!
    )
    
    # Apply LoRA to VLM for language training
    print("Applying LoRA to VLM...")
    vlm_lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=training_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    base_model.vlm = get_peft_model(base_model.vlm, vlm_lora_config)
    base_model.vlm.print_trainable_parameters()
    
    # Apply LoRA to Expert for trajectory conditioning
    print("Applying LoRA to Expert...")
    expert_lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=training_args.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    base_model.expert = get_peft_model(base_model.expert, expert_lora_config)
    base_model.expert.print_trainable_parameters()
    
    # Train projection layers (critical for trajectory)
    for param in base_model.action_in_proj.parameters():
        param.requires_grad = True
    for param in base_model.action_out_proj.parameters():
        param.requires_grad = True
    for param in base_model.diffusion.parameters():
        param.requires_grad = False
    
    # Wrap in our training model
    model = AlpamayoFullModel(base_model, traj_loss_weight=training_args.traj_loss_weight)
    
    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.vlm.enable_input_require_grads()
    
    # Print trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Create dataset
    processor = helper.get_processor(base_model.tokenizer)
    dataset = FullPipelineDataset(
        data_path=data_args.data_path,
        processor=processor,
        max_length=training_args.max_length
    )
    
    # Create DataLoader (no Trainer!)
    from torch.utils.data import DataLoader
    
    def custom_collate(batch):
        return collate_fn(batch, base_model.tokenizer)
    
    dataloader = DataLoader(
        dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=0,  # For debug
    )
    
    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_args.learning_rate,
    )
    
    # Training loop
    model.train()
    accumulation_steps = training_args.gradient_accumulation_steps
    global_step = 0
    total_steps = len(dataloader) * training_args.num_train_epochs // accumulation_steps
    
    print(f"Starting training: {total_steps} steps, {training_args.num_train_epochs} epochs")
    
    from tqdm import tqdm
    
    for epoch in range(int(training_args.num_train_epochs)):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        accumulated_loss = 0.0
        
        for step, batch in enumerate(pbar):
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                gt_trajectory=batch.get("gt_trajectory"),
                ego_history_xyz=batch.get("ego_history_xyz"),
                ego_history_rot=batch.get("ego_history_rot"),
            )
            
            loss = outputs["loss"] / accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()
            
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                pbar.set_postfix({"loss": accumulated_loss * accumulation_steps})
                accumulated_loss = 0.0
                
                # Save checkpoint
                if global_step % training_args.save_steps == 0:
                    save_path = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    torch.save({
                        "expert_lora": model.expert.state_dict(),
                        "action_in_proj": model.action_in_proj.state_dict(),
                        "action_out_proj": model.action_out_proj.state_dict(),
                        "step": global_step,
                    }, os.path.join(save_path, "expert_diffusion.pt"))
                    print(f"\nSaved checkpoint at step {global_step}")
    
    # Save final model
    os.makedirs(training_args.output_dir, exist_ok=True)
    torch.save({
        "expert_lora": model.expert.state_dict(),
        "action_in_proj": model.action_in_proj.state_dict(),
        "action_out_proj": model.action_out_proj.state_dict(),
    }, os.path.join(training_args.output_dir, "expert_diffusion_final.pt"))
    
    print("Training complete!")


if __name__ == "__main__":
    train()

