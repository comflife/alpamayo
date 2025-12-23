#!/usr/bin/env python3
"""
Consistency-Enhanced Fine-tuning for Alpamayo-R1.

Adds multi-modal consistency losses on top of standard SFT:
- Vision-Language consistency (VLM already does this)
- Vision-Trajectory consistency (NEW)
- Language-Trajectory consistency (NEW)

This enforces that reasoning, vision, and trajectory are semantically aligned,
addressing the limitation where SFT only learns coordinate-level trajectory
matching without understanding the semantic meaning.

Usage:
cd /home/byounggun/alpamayo/src
torchrun --nproc_per_node=2 -m alpamayo_r1.alignment.finetune_consistency \
    --data_path /home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset/finetune_data.jsonl \
    --output_dir /home/byounggun/alpamayo/outputs/alpamayo_consistency_finetuned \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --consistency_loss_weight 0.1
"""

import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import numpy as np
from PIL import Image

sys.path.insert(0, '/home/byounggun/alpamayo/src')

import transformers
from transformers import Trainer, TrainingArguments, HfArgumentParser
from peft import LoraConfig, get_peft_model, PeftModel

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="nvidia/Alpamayo-R1-10B")


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to JSONL training data"})


@dataclass
class ConsistencyTrainingArguments(TrainingArguments):
    max_length: int = field(default=512)
    
    # LoRA
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    
    # Loss weighting
    traj_loss_weight: float = field(default=1.0)
    consistency_loss_weight: float = field(default=0.1, metadata={"help": "Weight for consistency loss"})
    
    # Consistency temperature
    consistency_temperature: float = field(default=0.07, metadata={"help": "Temperature for contrastive loss"})
    
    # Memory optimization
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = True


# ==============================================================================
# Consistency Loss Functions
# ==============================================================================

def compute_multimodal_consistency_loss(
    vision_features: torch.Tensor,      # (B, D)
    language_features: torch.Tensor,    # (B, D)
    trajectory_features: torch.Tensor,  # (B, D)
    temperature: float = 0.07,
) -> Dict[str, torch.Tensor]:
    """
    Compute contrastive consistency losses between modalities.
    Uses InfoNCE (CLIP-style) to enforce alignment.
    """
    B = vision_features.shape[0]
    
    # Normalize
    vision_features = F.normalize(vision_features, dim=-1)
    language_features = F.normalize(language_features, dim=-1)
    trajectory_features = F.normalize(trajectory_features, dim=-1)
    
    # Similarity matrices (B, B) - diagonal = positive pairs
    sim_vl = torch.matmul(vision_features, language_features.T) / temperature
    sim_vt = torch.matmul(vision_features, trajectory_features.T) / temperature
    sim_lt = torch.matmul(language_features, trajectory_features.T) / temperature
    
    # Targets: diagonal elements are positive
    targets = torch.arange(B, device=vision_features.device)
    
    # InfoNCE loss (both directions)
    vl_loss = (F.cross_entropy(sim_vl, targets) + F.cross_entropy(sim_vl.T, targets)) / 2
    vt_loss = (F.cross_entropy(sim_vt, targets) + F.cross_entropy(sim_vt.T, targets)) / 2
    lt_loss = (F.cross_entropy(sim_lt, targets) + F.cross_entropy(sim_lt.T, targets)) / 2
    
    total = vl_loss + vt_loss + lt_loss
    
    return {
        "vl_loss": vl_loss,
        "vt_loss": vt_loss,
        "lt_loss": lt_loss,
        "total_consistency_loss": total,
    }


def extract_vision_features(model, pixel_values, image_grid_thw):
    """Extract vision features from VLM's vision tower."""
    # Access vision tower
    if hasattr(model, 'visual'):
        vision_tower = model.visual
    elif hasattr(model, 'model') and hasattr(model.model, 'visual'):
        vision_tower = model.model.visual
    elif hasattr(model, 'base_model'):
        if hasattr(model.base_model, 'visual'):
            vision_tower = model.base_model.visual
        elif hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'visual'):
            vision_tower = model.base_model.model.visual
    else:
        raise AttributeError("Cannot find vision tower")
    
    # Forward
    vision_outputs = vision_tower(pixel_values, grid_thw=image_grid_thw)
    # Pool: (B, num_patches, D) -> (B, D)
    vision_features = vision_outputs.mean(dim=1)
    
    return vision_features


def extract_language_features(model, input_ids, attention_mask):
    """Extract language features from VLM."""
    # Get hidden states
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    
    # Mean pooling with mask
    hidden = outputs.hidden_states[-1]  # (B, seq_len, D)
    mask = attention_mask.unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    
    return pooled


def extract_trajectory_features(expert_model, trajectory):
    """
    Extract trajectory features from Expert model.
    
    Args:
        expert_model: The expert transformer
        trajectory: (B, T, 3) - trajectory waypoints
    
    Returns:
        (B, D) - trajectory embeddings
    """
    B, T, _ = trajectory.shape
    
    # Create simple positional encoding
    pos_enc = torch.arange(T, device=trajectory.device).float().unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    pos_enc = pos_enc.expand(B, T, 1) / T  # Normalize
    
    # Concatenate trajectory with time
    traj_with_time = torch.cat([trajectory, pos_enc], dim=-1)  # (B, T, 4)
    
    # Forward through expert (treating as sequence)
    expert_outputs = expert_model(
        inputs_embeds=traj_with_time,
        output_hidden_states=True,
        return_dict=True,
    )
    
    # Pool over time dimension
    hidden = expert_outputs.hidden_states[-1]  # (B, T, D)
    pooled = hidden.mean(dim=1)  # (B, D)
    
    return pooled


# ==============================================================================
# Training Model
# ==============================================================================

class ConsistencyEnhancedModel(nn.Module):
    """Alpamayo with multi-modal consistency."""
    
    def __init__(
        self,
        base_model: AlpamayoR1,
        traj_loss_weight: float = 1.0,
        consistency_loss_weight: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.base_model = base_model
        self.traj_loss_weight = traj_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.temperature = temperature
        
        # Components
        self.vlm = base_model.vlm
        self.expert = base_model.expert
        self.diffusion = base_model.diffusion
        self.action_space = base_model.action_space
        self.action_in_proj = base_model.action_in_proj
        self.action_out_proj = base_model.action_out_proj
        self.config = base_model.config
    
    @property
    def tokenizer(self):
        return self.base_model.tokenizer
    
    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        pixel_values=None,
        image_grid_thw=None,
        gt_trajectory=None,
        ego_history_xyz=None,
        ego_history_rot=None,
        **kwargs,
    ):
        device = input_ids.device
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        loss_dict = {}
        
        # 1. Language Loss
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
        
        if vlm_outputs.loss is not None:
            total_loss = total_loss + vlm_outputs.loss
            loss_dict["language_loss"] = vlm_outputs.loss.item()
        
        # 2. Trajectory Loss (Flow Matching)
        if gt_trajectory is not None and ego_history_xyz is not None:
            traj_loss = self._compute_trajectory_loss(
                gt_trajectory, ego_history_xyz, ego_history_rot
            )
            total_loss = total_loss + self.traj_loss_weight * traj_loss
            loss_dict["traj_loss"] = traj_loss.item()
        
        # 3. **Consistency Loss (NEW!)**
        if pixel_values is not None and gt_trajectory is not None and self.consistency_loss_weight > 0:
            try:
                # Extract features
                vision_feats = extract_vision_features(
                    self.vlm, pixel_values, image_grid_thw
                )
                language_feats = extract_language_features(
                    self.vlm, input_ids, attention_mask
                )
                trajectory_feats = extract_trajectory_features(
                    self.expert, gt_trajectory
                )
                
                # Compute consistency
                consistency_losses = compute_multimodal_consistency_loss(
                    vision_feats, language_feats, trajectory_feats,
                    temperature=self.temperature,
                )
                
                consistency_loss = consistency_losses["total_consistency_loss"]
                total_loss = total_loss + self.consistency_loss_weight * consistency_loss
                
                loss_dict["consistency_loss"] = consistency_loss.item()
                loss_dict["vl_loss"] = consistency_losses["vl_loss"].item()
                loss_dict["vt_loss"] = consistency_losses["vt_loss"].item()
                loss_dict["lt_loss"] = consistency_losses["lt_loss"].item()
            except Exception as e:
                print(f"Warning: Consistency loss failed: {e}")
        
        return {
            "loss": total_loss,
            "loss_dict": loss_dict,
        }
    
    def _compute_trajectory_loss(self, gt_trajectory, ego_history_xyz, ego_history_rot):
        """Compute flow matching loss (같은 로직 유지)."""
        B, T, _ = gt_trajectory.shape
        device = gt_trajectory.device
        
        # Random timestep for flow matching
        t = torch.rand(B, device=device)
        
        # Sample noise
        noise = torch.randn_like(gt_trajectory)
        
        # Interpolate (flow matching)
        noisy_traj = t.view(-1, 1, 1) * gt_trajectory + (1 - t.view(-1, 1, 1)) * noise
        
        # Prepare action input
        action_input = self.action_in_proj(noisy_traj)
        
        # Diffusion prediction
        pred_traj = self.diffusion(
            x=action_input,
            t=t,
            ego_history_xyz=ego_history_xyz[:, 0],  # (B, 16, 3)
            ego_history_rot=ego_history_rot[:, 0],  # (B, 16, 3, 3)
        )
        
        # Flow matching loss
        target = gt_trajectory - noise
        loss = F.mse_loss(pred_traj, target)
        
        return loss


# ==============================================================================
# Dataset (same as finetune_full.py)
# ==============================================================================

class FullPipelineDataset(torch.utils.data.Dataset):
    """Dataset for VLM + Trajectory training."""
    
    def __init__(self, data_path, processor, max_length=512):
        self.data_path = data_path
        self.processor = processor
        self.max_length = max_length
        
        # Load data
        with open(data_path, "r") as f:
            self.samples = [json.loads(line) for line in f]
        
        print(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load frames
        frame_paths = sample['frame_paths'][:4]
        frames = []
        for fp in frame_paths:
            if os.path.exists(fp):
                img = Image.open(fp).convert("RGB")
                img_np = np.array(img)
                img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                frames.append(img_t)
        
        frames_tensor = torch.stack(frames, dim=0)
        
        # Create message
        reasoning = sample['reasoning']
        messages = helper.create_message(frames_tensor)
        messages.append({"role": "assistant", "content": reasoning})
        
        # Tokenize
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Ego history
        num_history = 16
        dt = 0.1
        speed = 5.0
        times = np.arange(-num_history + 1, 1) * dt
        positions = np.zeros((num_history, 3))
        positions[:, 0] = times * speed
        
        ego_history_xyz = torch.from_numpy(positions).float().unsqueeze(0)  # (1, 16, 3)
        ego_history_rot = torch.eye(3).unsqueeze(0).repeat(num_history, 1, 1).unsqueeze(0)  # (1, 16, 3, 3)
        
        # GT trajectory
        gt_traj = torch.tensor(sample['trajectory'], dtype=torch.float32)
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": inputs["input_ids"].squeeze(0).clone(),
            "pixel_values": inputs.get("pixel_values").squeeze(0) if "pixel_values" in inputs else None,
            "image_grid_thw": inputs.get("image_grid_thw").squeeze(0) if "image_grid_thw" in inputs else None,
            "gt_trajectory": gt_traj,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }


def collate_fn(batch, tokenizer):
    """Collate with padding."""
    # Pad sequences
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    max_len = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    for ids, mask, lab in zip(input_ids, attention_mask, labels):
        pad_len = max_len - len(ids)
        padded_input_ids.append(torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id)]))
        padded_attention_mask.append(torch.cat([mask, torch.zeros(pad_len)]))
        padded_labels.append(torch.cat([lab, torch.full((pad_len,), -100)]))
    
    result = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels),
    }
    
    # Stack other fields
    for key in ["pixel_values", "image_grid_thw", "gt_trajectory", "ego_history_xyz", "ego_history_rot"]:
        if key in batch[0] and batch[0][key] is not None:
            vals = [item[key] for item in batch]
            if all(v is not None for v in vals):
                result[key] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    
    return result


# ==============================================================================
# Training Loop
# ==============================================================================

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, ConsistencyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Rank {rank}/{world_size}] Using DDP on device {device}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Running on single device: {device}")
    
    # Load base model
    if rank == 0:
        print(f"Loading Alpamayo model from {model_args.model_name_or_path}...")
    
    base_model = AlpamayoR1.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.bfloat16,
    )
    base_model = base_model.to(device)
    
    # Apply LoRA to VLM
    if rank == 0:
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
    if rank == 0:
        base_model.vlm.print_trainable_parameters()
    
    # Apply LoRA to Expert
    if rank == 0:
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
    if rank == 0:
        base_model.expert.print_trainable_parameters()
    
    # Train projection layers
    for param in base_model.action_in_proj.parameters():
        param.requires_grad = True
    for param in base_model.action_out_proj.parameters():
        param.requires_grad = True
    for param in base_model.diffusion.parameters():
        param.requires_grad = False
    
    # Wrap in consistency model
    model = ConsistencyEnhancedModel(
        base_model,
        traj_loss_weight=training_args.traj_loss_weight,
        consistency_loss_weight=training_args.consistency_loss_weight,
        temperature=training_args.consistency_temperature,
    )
    
    # Gradient checkpointing
    if training_args.gradient_checkpointing:
        model.vlm.enable_input_require_grads()
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if rank == 0:
            print(f"Model wrapped with DDP")
    
    # Dataset
    processor = helper.get_processor(base_model.tokenizer)
    dataset = FullPipelineDataset(
        data_path=data_args.data_path,
        processor=processor,
        max_length=training_args.max_length
    )
    
    # DataLoader
    from torch.utils.data import DataLoader
    
    def custom_collate(batch):
        return collate_fn(batch, base_model.tokenizer)
    
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        dataloader = DataLoader(
            dataset, batch_size=training_args.per_device_train_batch_size,
            sampler=sampler, collate_fn=custom_collate, num_workers=4,
            pin_memory=training_args.dataloader_pin_memory,
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=training_args.per_device_train_batch_size,
            shuffle=True, collate_fn=custom_collate, num_workers=4,
            pin_memory=training_args.dataloader_pin_memory,
        )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_args.learning_rate,
    )
    
    # Training loop
    model.train()
    accumulation_steps = training_args.gradient_accumulation_steps
    global_step = 0
    
    if rank == 0:
        print(f"Starting training: {training_args.num_train_epochs} epochs")
    
    from tqdm import tqdm
    
    for epoch in range(int(training_args.num_train_epochs)):
        if world_size > 1:
            sampler.set_epoch(epoch)
        
        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        else:
            pbar = dataloader
        
        accumulated_loss = 0.0
        
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
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
                
                if rank == 0:
                    loss_dict = outputs.get("loss_dict", {})
                    pbar.set_postfix({
                        "loss": accumulated_loss * accumulation_steps,
                        **{k: f"{v:.4f}" for k, v in loss_dict.items()}
                    })
                accumulated_loss = 0.0
                
                # Save checkpoint
                if rank == 0 and global_step % training_args.save_steps == 0:
                    save_path = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    
                    model_to_save = model.module if isinstance(model, DDP) else model
                    
                    # Save VLM LoRA
                    model_to_save.vlm.save_pretrained(os.path.join(save_path, "vlm_lora"))
                    
                    # Save Expert + Diffusion decoder
                    if global_step % (training_args.save_steps * 3) == 0:
                        torch.save({
                            "expert_lora": model_to_save.expert.state_dict(),
                            "action_in_proj": model_to_save.action_in_proj.state_dict(),
                            "action_out_proj": model_to_save.action_out_proj.state_dict(),
                            "step": global_step,
                        }, os.path.join(save_path, "expert_diffusion.pt"))
                        print(f"\nSaved full checkpoint at step {global_step}")
                    else:
                        print(f"\nSaved VLM LoRA at step {global_step}")
                
                if world_size > 1:
                    dist.barrier()
    
    # Save final
    if rank == 0:
        final_save_path = os.path.join(training_args.output_dir, "final")
        os.makedirs(final_save_path, exist_ok=True)
        
        model_to_save = model.module if isinstance(model, DDP) else model
        model_to_save.vlm.save_pretrained(os.path.join(final_save_path, "vlm_lora"))
        torch.save({
            "expert_lora": model_to_save.expert.state_dict(),
            "action_in_proj": model_to_save.action_in_proj.state_dict(),
            "action_out_proj": model_to_save.action_out_proj.state_dict(),
        }, os.path.join(final_save_path, "expert_diffusion.pt"))
        
        print("Training complete!")
    
    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    train()
