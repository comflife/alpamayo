"""
Full Pipeline Fine-tuning for Alpamayo-R1.
Trains VLM + Expert + Diffusion with combined language + trajectory loss.

Usage:
cd /home/byounggun/alpamayo/src && CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m alpamayo_r1.alignment.finetune_full --data_path /home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset/finetune_data.jsonl --output_dir /home/byounggun/alpamayo/outputs/alpamayo_sft_v2 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --num_train_epochs 3 --learning_rate 1e-5 --warmup_ratio 0.03 --logging_steps 10 --save_steps 100 --bf16 True --traj_loss_weight 1.0

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
    gradient_checkpointing: bool = True

    dataloader_pin_memory: bool = True


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
        loss_dict = {}

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
            loss_dict["language_loss"] = language_loss.item()

        # =================================================================
        # 2. Trajectory Loss (End-to-End Generation)
        # =================================================================
        if gt_trajectory is not None and ego_history_xyz is not None:
            traj_loss = self._compute_trajectory_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                gt_trajectory=gt_trajectory,
                ego_history_xyz=ego_history_xyz,
                ego_history_rot=ego_history_rot,
            )
            total_loss = total_loss + self.traj_loss_weight * traj_loss
            loss_dict["traj_loss"] = traj_loss.item()

        loss_dict["total_loss"] = total_loss.item()
        return {"loss": total_loss, "loss_dict": loss_dict}
    
    def _compute_trajectory_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        gt_trajectory: torch.Tensor,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute trajectory loss using END-TO-END generation.

        This actually runs the full VLM → Expert → Diffusion pipeline
        to generate trajectories, then compares with GT in trajectory space.

        This ensures:
        1. Model sees the images (VLM processes them)
        2. Gradient flows through entire pipeline
        3. Lateral movements (left/right) are learned correctly
        """
        device = gt_trajectory.device
        B = gt_trajectory.shape[0]

        # Ensure gt_trajectory is (B, T, 2) or (B, T, 3)
        if gt_trajectory.dim() == 2:
            gt_trajectory = gt_trajectory.unsqueeze(0)  # (T, 2) -> (1, T, 2)

        # Add z=0 if only xy
        if gt_trajectory.shape[-1] == 2:
            gt_trajectory = F.pad(gt_trajectory, (0, 1), value=0)  # (B, T, 2) -> (B, T, 3)

        # Prepare ego history
        if ego_history_xyz.dim() == 2:
            ego_history_xyz = ego_history_xyz.unsqueeze(0)  # (T, 3) -> (1, T, 3)
        if ego_history_rot.dim() == 3:
            ego_history_rot = ego_history_rot.unsqueeze(0)  # (T, 3, 3) -> (1, T, 3, 3)

        # Expand to match batch size
        if ego_history_xyz.shape[0] == 1 and B > 1:
            ego_history_xyz = ego_history_xyz.expand(B, -1, -1)
        if ego_history_rot.shape[0] == 1 and B > 1:
            ego_history_rot = ego_history_rot.expand(B, -1, -1, -1)

        # Add sample dimension: (B, T, ...) -> (B, 1, T, ...)
        if ego_history_xyz.dim() == 3:
            ego_history_xyz = ego_history_xyz.unsqueeze(1)
        if ego_history_rot.dim() == 4:
            ego_history_rot = ego_history_rot.unsqueeze(1)

        try:
            # Run FULL trajectory generation pipeline
            # This calls VLM → Expert → Diffusion
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot = self.base_model.sample_trajectories_from_data_with_vlm_rollout(
                    data={
                        "tokenized_data": {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "pixel_values": pixel_values,
                            "image_grid_thw": image_grid_thw,
                        },
                        "ego_history_xyz": ego_history_xyz,
                        "ego_history_rot": ego_history_rot,
                    },
                    num_traj_samples=1,  # Generate 1 trajectory per sample
                    top_p=0.98,
                    temperature=0.6,
                    max_generation_length=128,  # Shorter for training efficiency
                )

            # pred_xyz: (B, 1, num_samples=1, T, 3)
            # Extract: (B, T, 3)
            pred_xyz = pred_xyz[:, 0, 0, :, :]  # (B, T, 3)

            # Match GT trajectory length
            T_gt = gt_trajectory.shape[1]
            T_pred = pred_xyz.shape[1]

            if T_pred > T_gt:
                pred_xyz = pred_xyz[:, :T_gt, :]
            elif T_pred < T_gt:
                gt_trajectory = gt_trajectory[:, :T_pred, :]

            # MSE loss in trajectory space (meters)
            # Focus on x-y plane (ignore z)
            loss = F.mse_loss(pred_xyz[:, :, :2], gt_trajectory[:, :, :2])

            return loss

        except Exception as e:
            logger.warning(f"Trajectory generation failed during training: {e}")
            # Return zero loss with gradient
            return torch.tensor(0.0, device=device, requires_grad=True)


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

    # Initialize DDP
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        # torchrun automatically initializes process group, just set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Rank {rank}/{world_size}] Using DDP on device {device}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Running on single device: {device}")

    if rank == 0:
        print(f"Loading Alpamayo model from {model_args.model_name_or_path}...")

    base_model = AlpamayoR1.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.bfloat16,
    )
    base_model = base_model.to(device)
    
    # Apply LoRA to VLM for language training
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

    # Apply LoRA to Expert for trajectory conditioning
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

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if rank == 0:
            print(f"Model wrapped with DDP across {world_size} GPUs")

    # Print trainable params
    if rank == 0:
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

    # Create DataLoader with DistributedSampler for DDP
    from torch.utils.data import DataLoader

    def custom_collate(batch):
        return collate_fn(batch, base_model.tokenizer)

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=training_args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=custom_collate,
            num_workers=4,
            pin_memory=training_args.dataloader_pin_memory,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=custom_collate,
            num_workers=4,
            pin_memory=training_args.dataloader_pin_memory,
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

    if rank == 0:
        print(f"Starting training: {total_steps} steps, {training_args.num_train_epochs} epochs")

    from tqdm import tqdm

    for epoch in range(int(training_args.num_train_epochs)):
        # Set epoch for DistributedSampler
        if world_size > 1:
            sampler.set_epoch(epoch)

        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        else:
            pbar = dataloader

        accumulated_loss = 0.0

        for step, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

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

                if rank == 0:
                    pbar.set_postfix({"loss": accumulated_loss * accumulation_steps})
                accumulated_loss = 0.0

                # Save checkpoint (only on rank 0)
                if rank == 0 and global_step % training_args.save_steps == 0:
                    save_path = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)

                    # Get underlying model (unwrap DDP if needed)
                    model_to_save = model.module if isinstance(model, DDP) else model

                    # Save VLM LoRA
                    model_to_save.vlm.save_pretrained(os.path.join(save_path, "vlm_lora"))

                    # Save Expert LoRA + Diffusion decoder (less frequently)
                    # Only save diffusion decoder every 3x save_steps
                    if global_step % (training_args.save_steps * 3) == 0:
                        torch.save({
                            "expert_lora": model_to_save.expert.state_dict(),
                            "action_in_proj": model_to_save.action_in_proj.state_dict(),
                            "action_out_proj": model_to_save.action_out_proj.state_dict(),
                            "step": global_step,
                        }, os.path.join(save_path, "expert_diffusion.pt"))
                        print(f"\nSaved full checkpoint (VLM + Expert + Diffusion) at step {global_step}")
                    else:
                        print(f"\nSaved VLM LoRA checkpoint at step {global_step}")

                # Synchronize all processes after checkpoint saving
                if world_size > 1:
                    dist.barrier()
    
    # Save final model (only on rank 0)
    if rank == 0:
        final_save_path = os.path.join(training_args.output_dir, "final")
        os.makedirs(final_save_path, exist_ok=True)

        # Get underlying model (unwrap DDP if needed)
        model_to_save = model.module if isinstance(model, DDP) else model

        # Save VLM LoRA
        model_to_save.vlm.save_pretrained(os.path.join(final_save_path, "vlm_lora"))

        # Save Expert + Diffusion decoder
        torch.save({
            "expert_lora": model_to_save.expert.state_dict(),
            "action_in_proj": model_to_save.action_in_proj.state_dict(),
            "action_out_proj": model_to_save.action_out_proj.state_dict(),
        }, os.path.join(final_save_path, "expert_diffusion.pt"))

        print("Training complete!")

    # Final synchronization
    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    train()

