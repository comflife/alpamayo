#!/usr/bin/env python3
"""
Self-Reflective Denoising RL (SRD-RL) for Alpamayo-R1.

════════════════════════════════════════════════════════════════════════════════
                    🎯 RESEARCH CONTRIBUTION: Learning from Noisy Labels
════════════════════════════════════════════════════════════════════════════════

PROBLEM: Traditional SFT blindly mimics GT labels, even when they're wrong.
If GT says "drive into mud", the model learns dangerous behavior.

SOLUTION: Teach the model to QUESTION the GT using visual evidence and reasoning.

════════════════════════════════════════════════════════════════════════════════
                              🧠 CORE INNOVATIONS
════════════════════════════════════════════════════════════════════════════════

1. **Trust Gate (Dynamic GT Weighting)**
   ----------------------------------------
   The model computes a "trust score" for each GT label:

   IF (reasoning mentions "danger/mud/obstacle") AND (visual_safety_score < 0.3):
       gt_weight = 0.1  # DISTRUST GT! Model thinks GT is wrong.
   ELSE:
       gt_weight = 1.0  # Trust GT normally

   This allows the model to OVERRIDE bad labels when evidence suggests they're harmful.

2. **Visual Safety Scoring (No Depth Required!)**
   ------------------------------------------------
   We use RGB texture analysis to detect dangerous terrain:

   - Laplacian Variance: Rough terrain (mud, rocks) has high texture complexity
   - Color Consistency: Compare path colors with safe reference area (hood/immediate road)
   - Combined Score: safety_score ∈ [0, 1], where 1 = safe, 0 = dangerous

   Example:
   - Smooth asphalt: Low variance, consistent color → Safety = 0.9
   - Mud pit: High variance, divergent color → Safety = 0.2

3. **Reasoning-Action Consistency**
   ----------------------------------
   Language and trajectory must be logically aligned:

   - If reasoning says "avoid left", trajectory should deviate RIGHT
   - If reasoning says "straight", trajectory should have low lateral deviation
   - Misalignment is penalized

4. **GRPO-style Multi-Sample Learning**
   --------------------------------------
   For each batch:
   1. Sample N trajectory variants (GT + noise-perturbed versions)
   2. Compute reward for each: R = w_safety * safety + w_gt * similarity + w_reasoning * alignment
   3. Compute advantage: A = R - baseline
   4. Policy gradient: Maximize log_prob of high-reward trajectories

════════════════════════════════════════════════════════════════════════════════
                           📊 EXPECTED OUTCOMES
════════════════════════════════════════════════════════════════════════════════

After training, the model will:
✓ Detect errors in GT labels using visual evidence
✓ Generate safer trajectories than human-labeled data
✓ Provide explainable reasoning ("GT says straight, but I see mud, so I'll detour")
✓ Achieve "super-human" performance on noisy datasets

════════════════════════════════════════════════════════════════════════════════
                                 🚀 USAGE
════════════════════════════════════════════════════════════════════════════════

Basic training:
---------------
cd /home/byounggun/alpamayo/src
torchrun --nproc_per_node=2 -m alpamayo_r1.alignment.finetune_consistency \
    --data_path /path/to/finetune_data.jsonl \
    --output_dir /path/to/output \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --learning_rate 5e-6

Key hyperparameters:
--------------------
--safety_reward_weight 1.5       # Higher = prioritize visual safety over GT
--gt_reward_weight 0.5           # Lower = less trust in GT labels
--reasoning_reward_weight 0.3    # Weight for language-action consistency
--num_trajectory_samples 4       # More samples = better RL exploration
--rl_loss_weight 0.5             # Balance between SFT and RL
--danger_keyword_threshold 0.3   # Safety threshold for triggering distrust

Aggressive distrust mode (for very noisy datasets):
----------------------------------------------------
--safety_reward_weight 2.0 --gt_reward_weight 0.3 --gt_trust_min 0.05

Conservative mode (clean datasets):
-----------------------------------
--safety_reward_weight 0.5 --gt_reward_weight 1.0 --rl_loss_weight 0.2

════════════════════════════════════════════════════════════════════════════════
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
import inspect
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
from PIL import Image
import cv2

sys.path.insert(0, '/home/byounggun/alpamayo/src')

import transformers
from transformers import Trainer, TrainingArguments, HfArgumentParser
from peft import LoraConfig, get_peft_model, PeftModel

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="nvidia/Alpamayo-R1-10B")


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to JSONL training data"})


@dataclass
class ConsistencyTrainingArguments(TrainingArguments):
    # Qwen3-VL expands each image into many visual tokens; too-small max_length
    # can truncate inside those tokens and cause a processor mismatch error.
    max_length: int = field(default=2048)

    # LoRA
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)

    # Loss weighting
    traj_loss_weight: float = field(default=1.0)
    consistency_loss_weight: float = field(default=0.2, metadata={"help": "Weight for consistency loss"})

    # SRD-RL specific params
    safety_reward_weight: float = field(default=1.5, metadata={"help": "Weight for visual safety reward"})
    gt_reward_weight: float = field(default=0.5, metadata={"help": "Base weight for GT matching reward"})
    reasoning_reward_weight: float = field(default=0.3, metadata={"help": "Weight for reasoning consistency"})
    num_trajectory_samples: int = field(default=4, metadata={"help": "Number of trajectory samples for GRPO"})
    rl_loss_weight: float = field(default=0.5, metadata={"help": "Weight for RL loss vs SFT loss"})

    # Trust gate thresholds
    danger_keyword_threshold: float = field(default=0.3, metadata={"help": "Safety score threshold for danger detection"})
    gt_trust_min: float = field(default=0.1, metadata={"help": "Minimum GT trust weight"})
    gt_trust_max: float = field(default=1.0, metadata={"help": "Maximum GT trust weight"})

    # Consistency temperature
    consistency_temperature: float = field(default=0.07, metadata={"help": "Temperature for contrastive loss"})

    # Memory optimization
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = True


# ==============================================================================
# Visual Safety Scoring (No Depth Required!)
# ==============================================================================

def compute_texture_variance(image_tensor: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
    """
    Compute texture variance along trajectory path.

    Intuition: Safe roads have uniform texture (low variance).
               Mud, rocks, water have complex texture (high variance).

    Args:
        image_tensor: (B, C, H, W) - RGB images
        trajectory: (B, T, 2 or 3) - XY or XYZ waypoints in ego frame

    Returns:
        variance_score: (B,) - Lower is safer (0-1 normalized)
    """
    B, C, H, W = image_tensor.shape
    device = image_tensor.device

    # Convert to numpy for OpenCV processing
    variances = []

    for b in range(B):
        img = image_tensor[b].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        img = (img * 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Project trajectory to image coordinates
        # Assume simple camera model: X (forward) -> Y pixel, Y (lateral) -> X pixel
        traj_2d = trajectory[b, :, :2].cpu().numpy()  # (T, 2)

        # Normalize to image coordinates (rough projection)
        # X: 0-10m -> bottom to top (H)
        # Y: -5 to 5m -> left to right (W)
        x_coords = H - (traj_2d[:, 0] / 10.0 * H * 0.8).astype(int)  # Forward
        y_coords = (traj_2d[:, 1] / 5.0 * W / 2 + W / 2).astype(int)  # Lateral

        # Clip to image bounds
        x_coords = np.clip(x_coords, 0, H - 1)
        y_coords = np.clip(y_coords, 0, W - 1)

        # Extract patches along trajectory
        patch_size = 32
        patch_variances = []

        for x, y in zip(x_coords, y_coords):
            x1 = max(0, x - patch_size // 2)
            x2 = min(H, x + patch_size // 2)
            y1 = max(0, y - patch_size // 2)
            y2 = min(W, y + patch_size // 2)

            patch = gray[x1:x2, y1:y2]
            if patch.size > 0:
                # Compute Laplacian variance (edge/texture measure)
                laplacian = cv2.Laplacian(patch, cv2.CV_64F)
                variance = laplacian.var()
                patch_variances.append(variance)

        if patch_variances:
            avg_variance = np.mean(patch_variances)
        else:
            avg_variance = 0.0

        variances.append(avg_variance)

    # Normalize to 0-1 (using empirical max ~1000 for typical outdoor scenes)
    variances = torch.tensor(variances, device=device, dtype=torch.float32)
    normalized = torch.clamp(variances / 1000.0, 0, 1)

    return normalized


def compute_color_consistency(image_tensor: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
    """
    Check if trajectory path has consistent color with safe reference area.

    Safe reference: bottom center of image (hood/immediate road).
    Dangerous if path color diverges significantly.

    Returns:
        consistency_score: (B,) - Higher is safer (0-1)
    """
    B, C, H, W = image_tensor.shape
    device = image_tensor.device

    scores = []

    for b in range(B):
        img = image_tensor[b].permute(1, 2, 0).cpu().numpy()  # (H, W, C)

        # Reference patch: bottom center (safe road in front of vehicle)
        ref_y1 = int(H * 0.8)
        ref_y2 = H
        ref_x1 = int(W * 0.4)
        ref_x2 = int(W * 0.6)
        ref_patch = img[ref_y1:ref_y2, ref_x1:ref_x2]
        ref_color = ref_patch.mean(axis=(0, 1))  # (C,)

        # Trajectory path colors
        traj_2d = trajectory[b, :, :2].cpu().numpy()
        x_coords = H - (traj_2d[:, 0] / 10.0 * H * 0.8).astype(int)
        y_coords = (traj_2d[:, 1] / 5.0 * W / 2 + W / 2).astype(int)
        x_coords = np.clip(x_coords, 0, H - 1)
        y_coords = np.clip(y_coords, 0, W - 1)

        # Sample colors along path
        path_colors = []
        for x, y in zip(x_coords, y_coords):
            color = img[x, y]
            path_colors.append(color)

        if path_colors:
            path_colors = np.array(path_colors)  # (T, C)
            # Color distance from reference
            color_dists = np.linalg.norm(path_colors - ref_color, axis=1)
            avg_dist = color_dists.mean()
            # Normalize (empirical max ~0.5 for RGB in [0,1])
            consistency = 1.0 - np.clip(avg_dist / 0.5, 0, 1)
        else:
            consistency = 1.0

        scores.append(consistency)

    return torch.tensor(scores, device=device, dtype=torch.float32)


def compute_visual_safety_score(
    image_tensor: torch.Tensor,
    trajectory: torch.Tensor,
) -> torch.Tensor:
    """
    Combined visual safety score from texture and color analysis.

    Returns:
        safety_score: (B,) in [0, 1], where 1 is safest
    """
    # Lower variance = safer
    texture_var = compute_texture_variance(image_tensor, trajectory)
    texture_safety = 1.0 - texture_var

    # Higher consistency = safer
    color_consistency = compute_color_consistency(image_tensor, trajectory)

    # Combine (equal weight)
    safety_score = 0.5 * texture_safety + 0.5 * color_consistency

    return safety_score


# ==============================================================================
# Reasoning Analysis
# ==============================================================================

def detect_danger_keywords(reasoning_text: str) -> Tuple[bool, float]:
    """
    Detect if reasoning mentions danger/obstacles.

    Returns:
        (has_danger, confidence)
    """
    danger_keywords = [
        r'\bmud\b', r'\brock\b', r'\bstone\b', r'\bobstacle\b',
        r'\bdanger\b', r'\bunsafe\b', r'\brough\b', r'\bbump\b',
        r'\bwater\b', r'\bpuddle\b', r'\bavoid\b', r'\bdetour\b',
        r'\bslippery\b', r'\buneven\b', r'\bhazard\b'
    ]

    text_lower = reasoning_text.lower()
    matches = 0

    for pattern in danger_keywords:
        if re.search(pattern, text_lower):
            matches += 1

    has_danger = matches > 0
    confidence = min(matches / 3.0, 1.0)  # Max at 3 keywords

    return has_danger, confidence


def check_reasoning_trajectory_alignment(
    reasoning_text: str,
    trajectory: torch.Tensor,
    gt_trajectory: torch.Tensor,
) -> float:
    """
    Check if reasoning explains the trajectory choice.

    E.g., if reasoning says "avoid left", trajectory should deviate right.

    Returns:
        alignment_score: 0-1
    """
    text_lower = reasoning_text.lower()

    # Compute trajectory deviation from GT
    traj_np = trajectory.cpu().numpy()  # (T, 2 or 3)
    gt_np = gt_trajectory.cpu().numpy()

    # Lateral deviation (Y axis)
    lateral_dev = (traj_np[:, 1] - gt_np[:, 1]).mean()

    # Check if language matches action
    if re.search(r'\bleft\b', text_lower):
        if lateral_dev > 0.5:  # Moved left
            return 1.0
        else:
            return 0.3
    elif re.search(r'\bright\b', text_lower):
        if lateral_dev < -0.5:  # Moved right
            return 1.0
        else:
            return 0.3
    elif re.search(r'\bstraight\b', text_lower):
        if abs(lateral_dev) < 0.5:
            return 1.0
        else:
            return 0.5
    else:
        # No directional keywords - neutral
        return 0.7


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
    # Compute in float32 to avoid dtype mismatch
    vision_features = vision_features.float()
    language_features = language_features.float()
    trajectory_features = trajectory_features.float()

    # Normalize
    vision_features = F.normalize(vision_features, dim=-1)
    language_features = F.normalize(language_features, dim=-1)
    trajectory_features = F.normalize(trajectory_features, dim=-1)
    
    # Gather features from all GPUs if using DDP
    if dist.is_available() and dist.is_initialized():
        # Function to gather tensors
        def gather_features(features):
            gathered = [torch.zeros_like(features) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, features)
            return torch.cat(gathered, dim=0)

        all_vision = gather_features(vision_features)
        all_language = gather_features(language_features)
        all_trajectory = gather_features(trajectory_features)
        
        local_B = vision_features.shape[0]
        rank = dist.get_rank()
        start_idx = rank * local_B
        
        # Targets for SigLIP: 1 for positive pair, 0 for others
        # Shape matches logits: (Local_B, Total_B)
        # Note: We assume equal batch sizes across ranks
        total_B = all_vision.shape[0]
        targets = torch.zeros(local_B, total_B, device=vision_features.device)
        
        # Set diagonal (matching indices) to 1
        # Row i (local) matches Col (start_idx + i) (global)
        indices = torch.arange(local_B, device=vision_features.device)
        targets[indices, indices + start_idx] = 1.0
        
        # SigLIP Loss (BCEWithLogits) - pairwise independent
        # Replaces Softmax (competitive) with Sigmoid (independent)
        # Handles "similar negatives" better because they aren't forced to sum to 1
        
        temp = float(temperature)
        # Add a learned bias if we had one, but here just scale
        # SigLIP typically uses logit_scale (like 1/temp) and logit_bias.
        # We'll use 1/temp as scale.
        
        def siglip_loss(feat_local, feat_all):
            logits = torch.matmul(feat_local, feat_all.T) / temp
            return F.binary_cross_entropy_with_logits(logits, targets)

        vl_loss = siglip_loss(vision_features, all_language)
        vt_loss = siglip_loss(vision_features, all_trajectory)
        lt_loss = siglip_loss(language_features, all_trajectory)
        
    else:
        # Local only (fallback)
        B = vision_features.shape[0]
        targets = torch.eye(B, device=vision_features.device)
        temp = float(temperature)
        
        def siglip_loss(feat1, feat2):
            logits = torch.matmul(feat1, feat2.T) / temp
            return F.binary_cross_entropy_with_logits(logits, targets)
        
        vl_loss = siglip_loss(vision_features, language_features)
        vt_loss = siglip_loss(vision_features, trajectory_features)
        lt_loss = siglip_loss(language_features, trajectory_features)

    # Note: Traditional SigLIP sums losses from both directions (Image->Text, Text->Image)
    # But since BCE is symmetric on the matrix (Logits_BA = Logits_AB.T),
    # and Targets are symmetric (Identity), Loss(AB) == Loss(BA) IF the matrix is square.
    # In DDP (Rectangular: B x World_B), we only compute Local -> All.
    # The gradients flows to both Local and All (which are detached? No all_gather usually has grad in torch 2.x? 
    # Actually dist.all_gather usually does NOT support autograd backprop through the gather unless using functional collectives).
    # Standard practice with standard all_gather: Loss is computed on each GPU for its local rows against all cols.
    # Effectively sums up to full matrix coverage across GPUs.
    
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
    
    # Match vision tower dtype to avoid dtype mismatch
    try:
        vision_dtype = next(vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(dtype=vision_dtype)
    except StopIteration:
        # If no parameters, use as is
        pass
    
    # Forward
    vision_outputs = vision_tower(pixel_values, grid_thw=image_grid_thw)

    # Qwen3-VL's visual module may return:
    # - a Tensor: (N_img, N_patches, D)
    # - a tuple: (image_embeds, deepstack_image_embeds)
    # - a ModelOutput-like object
    if isinstance(vision_outputs, tuple):
        vision_outputs = vision_outputs[0]
    elif hasattr(vision_outputs, "last_hidden_state"):
        vision_outputs = vision_outputs.last_hidden_state

    if not isinstance(vision_outputs, torch.Tensor):
        raise TypeError(f"Unexpected vision tower output type: {type(vision_outputs)}")

    # Pool patches: (N_img, num_patches, D) -> (N_img, D)
    if vision_outputs.dim() == 3:
        return vision_outputs.mean(dim=1)
    return vision_outputs


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
    # Use same dtype as hidden states to avoid dtype mismatch
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
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
    """Alpamayo with Self-Reflective Denoising RL (SRD-RL)."""

    def __init__(
        self,
        base_model: AlpamayoR1,
        traj_loss_weight: float = 1.0,
        consistency_loss_weight: float = 0.2,
        temperature: float = 0.07,
        # SRD-RL params
        safety_reward_weight: float = 1.5,
        gt_reward_weight: float = 0.5,
        reasoning_reward_weight: float = 0.3,
        num_trajectory_samples: int = 4,
        rl_loss_weight: float = 0.5,
        danger_threshold: float = 0.3,
        gt_trust_min: float = 0.1,
        gt_trust_max: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.traj_loss_weight = traj_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.temperature = temperature

        # SRD-RL hyperparameters
        self.safety_reward_weight = safety_reward_weight
        self.gt_reward_weight = gt_reward_weight
        self.reasoning_reward_weight = reasoning_reward_weight
        self.num_trajectory_samples = num_trajectory_samples
        self.rl_loss_weight = rl_loss_weight
        self.danger_threshold = danger_threshold
        self.gt_trust_min = gt_trust_min
        self.gt_trust_max = gt_trust_max
        
        # Components
        self.vlm = base_model.vlm
        self.expert = base_model.expert
        self.diffusion = base_model.diffusion
        self.action_space = base_model.action_space
        self.action_in_proj = base_model.action_in_proj
        self.action_out_proj = base_model.action_out_proj
        self.config = base_model.config

        # Consistency Projections
        # Determine dimensions
        lang_dim = None
        # Try direct hidden_size first (standard for many models)
        if hasattr(base_model.vlm.config, "hidden_size"):
             lang_dim = base_model.vlm.config.hidden_size
             print(f"DEBUG: Found lang_dim from config.hidden_size: {lang_dim}")
        
        if lang_dim is None:
            try:
                lang_dim = base_model.vlm.config.text_config.hidden_size
                print(f"DEBUG: Found lang_dim from config.text_config.hidden_size: {lang_dim}")
            except AttributeError:
                pass

        if lang_dim is None:
             lang_dim = 2048
             print(f"DEBUG: Defaulting lang_dim to {lang_dim}")

        try:
            vision_dim = base_model.vlm.config.vision_config.hidden_size
        except AttributeError:
            vision_dim = 4096 # Default info
            
        # We assume trajectory features come from expert/LLM space, so traj_dim = lang_dim
        traj_dim = lang_dim
        
        # Force common_dim to match the largest feature dimension (Language/Vision output usually align)
        # Runtime observation: Vision=4096, Lang=4096, Traj=2048. Common should be 4096.
        common_dim = 4096
        if lang_dim is not None and lang_dim > common_dim:
            common_dim = lang_dim
        
        print(f"DEBUG: Final Projs - Common: {common_dim}. Using LazyLinear to infer input dims.")

        self.vision_proj = nn.LazyLinear(common_dim).to(device=base_model.device, dtype=base_model.dtype)
        self.lang_proj = nn.LazyLinear(common_dim).to(device=base_model.device, dtype=base_model.dtype)
        self.traj_proj = nn.LazyLinear(common_dim).to(device=base_model.device, dtype=base_model.dtype)
        
        # Initialize LazyLinear modules by running a dummy forward pass
        # We use small dummy inputs on the correct device/dtype
        with torch.no_grad():
            dummy_dev = base_model.device
            dummy_dtype = base_model.dtype
            
            # Vision dummy: (1, 4096) or whatever it receives
            # We don't know exact input dim but LazyLinear will figure it out on first call.
            # However, DDP requires params to be initialized BEFORE wrapping.
            # So we MUST pass data with CORRECT input dimension.
            # Based on logs: Vision=4096, Lang=4096, Traj=2048.
            self.vision_proj(torch.zeros(1, 4096, device=dummy_dev, dtype=dummy_dtype))
            self.lang_proj(torch.zeros(1, 4096, device=dummy_dev, dtype=dummy_dtype))
            self.traj_proj(torch.zeros(1, 2048, device=dummy_dev, dtype=dummy_dtype))
            
            # Reset parameters to ensure clean initialization after dummy pass
            # CRITICAL: Initialize Vision/Lang as IDENTITY to preserve pre-trained alignment!
            # Random init destroys the alignment and makes learning hard with small batch size.
            
            # Vision (4096 -> 4096)
            if self.vision_proj.weight.shape[0] == self.vision_proj.weight.shape[1]:
                nn.init.eye_(self.vision_proj.weight)
                nn.init.zeros_(self.vision_proj.bias)
            else:
                 nn.init.kaiming_normal_(self.vision_proj.weight)
                 nn.init.zeros_(self.vision_proj.bias)

            # Lang (4096 -> 4096)
            if self.lang_proj.weight.shape[0] == self.lang_proj.weight.shape[1]:
                nn.init.eye_(self.lang_proj.weight)
                nn.init.zeros_(self.lang_proj.bias)
            else:
                 nn.init.kaiming_normal_(self.lang_proj.weight)
                 nn.init.zeros_(self.lang_proj.bias)
            
            # Traj (2048 -> 4096) - Must be learned (Random Init)
            nn.init.kaiming_normal_(self.traj_proj.weight)
            nn.init.zeros_(self.traj_proj.bias)
    
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
        reasoning_text=None,  # NEW: reasoning for RL evaluation
        rl_image=None,  # [NEW] Accept RL image
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

        # 2. Trajectory Loss (Flow Matching) - Standard SFT component
        if gt_trajectory is not None and ego_history_xyz is not None:
            traj_loss = self._compute_trajectory_loss(
                gt_trajectory, ego_history_xyz, ego_history_rot
            )
            total_loss = total_loss + self.traj_loss_weight * traj_loss
            loss_dict["traj_loss"] = traj_loss.item()

        # 2b. **NEW: Self-Reflective RL Loss**
        # Sample multiple trajectories and compute trust-aware rewards
        # Check for rl_image instead of pixel_values for safety scoring
        if (
            gt_trajectory is not None
            and ego_history_xyz is not None
            and rl_image is not None  # [FIX] Use rl_image
            and self.training
            and self.rl_loss_weight > 0
        ):
            try:
                rl_loss, rl_metrics = self._compute_rl_loss(
                    rl_image=rl_image,  # [FIX] Pass rl_image
                    gt_trajectory=gt_trajectory,
                    ego_history_xyz=ego_history_xyz,
                    ego_history_rot=ego_history_rot,
                    reasoning_text=reasoning_text,
                )
                total_loss = total_loss + self.rl_loss_weight * rl_loss
                loss_dict["rl_loss"] = rl_loss.item()
                loss_dict.update(rl_metrics)
            except Exception as e:
                import traceback
                print(f"Warning: RL loss failed: {e}")
                traceback.print_exc()
        
        # 3. **Consistency Loss (NEW!)**
        if (
            pixel_values is not None
            and gt_trajectory is not None
            and ego_history_xyz is not None
            and self.consistency_loss_weight > 0
        ):
            try:
                # Extract features
                per_image_vision = extract_vision_features(self.vlm, pixel_values, image_grid_thw)
                # Qwen3-VL batches images flattened as (B * n_img, ...). Pool back to (B, D).
                B = input_ids.shape[0]
                if per_image_vision.shape[0] != B and per_image_vision.shape[0] % B == 0:
                    n_img = per_image_vision.shape[0] // B
                    vision_feats = per_image_vision.view(B, n_img, -1).mean(dim=1)
                else:
                    vision_feats = per_image_vision
                language_feats = extract_language_features(
                    self.vlm, input_ids, attention_mask
                )

                # Trajectory features: convert trajectory -> action space -> action_in_proj embeds.
                # (Avoid feeding raw XYZ into the expert, which expects hidden-size embeddings.)
                traj = gt_trajectory
                if traj.shape[-1] == 2:
                    traj = F.pad(traj, (0, 1), value=0)

                hist_xyz = ego_history_xyz
                hist_rot = ego_history_rot
                if isinstance(hist_xyz, torch.Tensor) and hist_xyz.dim() == 4:
                    hist_xyz = hist_xyz[:, 0]
                if isinstance(hist_rot, torch.Tensor) and hist_rot.dim() == 5:
                    hist_rot = hist_rot[:, 0]

                if hist_rot is None or (isinstance(hist_rot, torch.Tensor) and hist_rot.numel() == 0):
                    T_hist = hist_xyz.shape[1]
                    hist_rot = (
                        torch.eye(3, device=hist_xyz.device, dtype=hist_xyz.dtype)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .expand(hist_xyz.shape[0], T_hist, 3, 3)
                        .contiguous()
                    )

                T_future = traj.shape[1]
                future_rot = (
                    torch.eye(3, device=traj.device, dtype=traj.dtype)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(traj.shape[0], T_future, 3, 3)
                    .contiguous()
                )

                gt_action = self.action_space.traj_to_action(
                    traj_history_xyz=hist_xyz.float(),
                    traj_history_rot=hist_rot.float(),
                    traj_future_xyz=traj.float(),
                    traj_future_rot=future_rot.float(),
                ).to(dtype=traj.dtype)

                # Use a fixed timestep (t=1) to embed the clean action
                # Match action_in_proj dtype to avoid mismatch (e.g. if model is bf16)
                try:
                    proj_dtype = next(self.action_in_proj.parameters()).dtype
                except StopIteration:
                    proj_dtype = gt_action.dtype

                gt_action_proj = gt_action.to(dtype=proj_dtype)
                t_feat = torch.ones(gt_action_proj.shape[0], 1, 1, device=gt_action.device, dtype=proj_dtype)
                traj_embeds = self.action_in_proj(gt_action_proj, t_feat)
                if traj_embeds.dim() == 2:
                    n_tokens = self.action_space.get_action_space_dims()[0]
                    traj_embeds = traj_embeds.view(gt_action_proj.shape[0], n_tokens, -1)
                trajectory_feats = traj_embeds.mean(dim=1)

                # Ensure all features match projection layer dtype before projection
                proj_dtype = self.base_model.dtype
                vision_feats = vision_feats.to(dtype=proj_dtype)
                language_feats = language_feats.to(dtype=proj_dtype)
                trajectory_feats = trajectory_feats.to(dtype=proj_dtype)

                # Project all to common dimension
                vision_feats = self.vision_proj(vision_feats)
                language_feats = self.lang_proj(language_feats)
                trajectory_feats = self.traj_proj(trajectory_feats)

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
                import traceback
                print(f"Warning: Consistency loss failed: {e}")
                traceback.print_exc()
        
        return {
            "loss": total_loss,
            "loss_dict": loss_dict,
        }
    
    def _compute_rl_loss(
        self,
        rl_image: torch.Tensor,  # [FIX] Changed arg name
        gt_trajectory: torch.Tensor,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
        reasoning_text: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Self-Reflective Denoising RL loss using raw images for safety check.
        """
        B = gt_trajectory.shape[0]
        
        # [FIX 1] Get dtype from model weights
        try:
            param = next(self.action_in_proj.parameters())
            device = param.device
            dtype = param.dtype
        except StopIteration:
            device = gt_trajectory.device
            dtype = torch.bfloat16

        # [FIX 2] Explicitly cast inputs to model dtype
        gt_trajectory = gt_trajectory.to(device=device, dtype=dtype)
        if ego_history_xyz is not None:
            ego_history_xyz = ego_history_xyz.to(device=device, dtype=dtype)
        if ego_history_rot is not None:
            ego_history_rot = ego_history_rot.to(device=device, dtype=dtype)

        if ego_history_xyz.dim() == 4:
            ego_history_xyz = ego_history_xyz[:, 0]
        if ego_history_rot is not None and ego_history_rot.dim() == 5:
            ego_history_rot = ego_history_rot[:, 0]

        if ego_history_rot is None or ego_history_rot.numel() == 0:
            T_hist = ego_history_xyz.shape[1]
            ego_history_rot = (
                torch.eye(3, device=device, dtype=dtype)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(B, T_hist, 3, 3)
                .contiguous()
            )

        # rl_image is already (B, C, H, W) from collate_fn
        # Ensure it is on the correct device for processing
        rl_image = rl_image.to(device=device)
        
        # ============================================================================
        # Sample multiple trajectories
        # ============================================================================
        if gt_trajectory.shape[-1] == 2:
            gt_traj_3d = F.pad(gt_trajectory, (0, 1), value=0)
        else:
            gt_traj_3d = gt_trajectory

        T_future = gt_traj_3d.shape[1]
        gt_future_rot = (
            torch.eye(3, device=device, dtype=dtype)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(B, T_future, 3, 3)
            .clone()
        )

        try:
            gt_action = self.action_space.traj_to_action(
                traj_history_xyz=ego_history_xyz.float(),
                traj_history_rot=ego_history_rot.float(),
                traj_future_xyz=gt_traj_3d.float(),
                traj_future_rot=gt_future_rot.float(),
            ).to(dtype=dtype)
        except Exception as e:
            logger.warning(f"Failed to convert GT trajectory to action: {e}")
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        sampled_trajectories = []
        sampled_actions = []

        sampled_trajectories.append(gt_traj_3d.detach())
        sampled_actions.append(gt_action.detach())

        n_tokens = self.action_space.get_action_space_dims()[0]
        timesteps = [0.3, 0.5, 0.7] if self.num_trajectory_samples >= 4 else [0.5]

        for t_val in timesteps[: self.num_trajectory_samples - 1]:
            with torch.no_grad():
                t = torch.full((B, 1, 1), t_val, device=device, dtype=dtype)
                x0 = torch.randn(gt_action.shape, device=device, dtype=dtype)
                x_t = (1 - t) * x0 + t * gt_action

                action_embeds = self.action_in_proj(x_t.to(dtype=dtype), t.to(dtype=dtype))
                if action_embeds.dim() == 2:
                    action_embeds = action_embeds.view(B, n_tokens, -1)

                expert_out = self.expert(
                    inputs_embeds=action_embeds,
                    use_cache=False,
                    return_dict=True,
                )
                last_hidden = expert_out.last_hidden_state
                pred_v = self.action_out_proj(last_hidden)
                pred_v = pred_v.view(B, *self.action_space.get_action_space_dims())

                sampled_action = x_t + pred_v * (1 - t)

                # [FIX 3] Handle Tuple return from action_to_traj
                action_out = self.action_space.action_to_traj(
                    sampled_action,
                    traj_history_xyz=ego_history_xyz.float(),
                    traj_history_rot=ego_history_rot.float(),
                )
                
                # Check if output is tuple or dict
                if isinstance(action_out, tuple):
                    sampled_traj = action_out[0] # Assume first element is XYZ
                elif isinstance(action_out, dict):
                    sampled_traj = action_out['traj_future_xyz']
                else:
                    # Fallback or error
                    sampled_traj = action_out

                sampled_trajectories.append(sampled_traj.detach())
                sampled_actions.append(sampled_action.detach())

        # ============================================================================
        # Compute log_probs
        # ============================================================================
        gt_action = self.action_space.traj_to_action(
            traj_history_xyz=ego_history_xyz.float(),
            traj_history_rot=ego_history_rot.float(),
            traj_future_xyz=gt_traj_3d.float(),
            traj_future_rot=gt_future_rot.float(),
        ).to(dtype=dtype)

        log_probs = []
        all_actions = [gt_action.detach()] + sampled_actions

        for action in all_actions:
            t = torch.ones(B, 1, 1, device=device, dtype=dtype)
            n_tokens = self.action_space.get_action_space_dims()[0]
            
            action_embeds = self.action_in_proj(action.to(dtype=dtype), t.to(dtype=dtype))
            if action_embeds.dim() == 2:
                action_embeds = action_embeds.view(B, n_tokens, -1)

            with torch.no_grad():
                expert_out = self.expert(
                    inputs_embeds=action_embeds,
                    use_cache=False,
                    return_dict=True,
                )
                last_hidden = expert_out.last_hidden_state
                pred_v = self.action_out_proj(last_hidden).view(
                    B, *self.action_space.get_action_space_dims()
                )

            neg_log_prob = (pred_v ** 2).mean(dim=[1, 2])
            log_prob = -neg_log_prob
            log_probs.append(log_prob)

        # ============================================================================
        # Reward Computation
        # ============================================================================
        all_rewards = []
        safety_scores_list = []
        gt_similarity_list = []
        reasoning_alignment_list = []

        gt_traj_3d_float = gt_traj_3d.float()

        for traj in sampled_trajectories:
            traj_float = traj.float()

            # 1. Visual Safety (USE rl_image HERE)
            safety_scores = compute_visual_safety_score(rl_image, traj_float)

            # 2. GT Similarity
            gt_dist = torch.norm(traj_float - gt_traj_3d_float, dim=-1).mean(dim=1)
            gt_similarity = torch.exp(-gt_dist / 2.0)

            # 3. Reasoning Alignment
            if reasoning_text is not None and len(reasoning_text) == B:
                reasoning_scores = []
                for b in range(B):
                    score = check_reasoning_trajectory_alignment(
                        reasoning_text[b], traj_float[b], gt_traj_3d_float[b]
                    )
                    reasoning_scores.append(score)
                reasoning_alignment = torch.tensor(reasoning_scores, device=device, dtype=torch.float32)
            else:
                reasoning_alignment = torch.ones(B, device=device, dtype=torch.float32) * 0.7

            # Trust Gate
            gt_weights = torch.ones(B, device=device, dtype=torch.float32) * self.gt_reward_weight

            if reasoning_text is not None and len(reasoning_text) == B:
                for b in range(B):
                    has_danger, danger_conf = detect_danger_keywords(reasoning_text[b])
                    if has_danger and safety_scores[b] < self.danger_threshold:
                        gt_weights[b] = self.gt_trust_min + (1 - danger_conf) * (self.gt_trust_max - self.gt_trust_min)
                    else:
                        gt_weights[b] = self.gt_trust_max * self.gt_reward_weight

            reward = (
                self.safety_reward_weight * safety_scores +
                gt_weights * gt_similarity +
                self.reasoning_reward_weight * reasoning_alignment
            )

            all_rewards.append(reward)
            safety_scores_list.append(safety_scores.mean().item())
            gt_similarity_list.append(gt_similarity.mean().item())
            reasoning_alignment_list.append(reasoning_alignment.mean().item())

        all_rewards = torch.stack(all_rewards, dim=0)
        baseline = all_rewards.mean(dim=0, keepdim=True)
        advantages = all_rewards - baseline

        # ============================================================================
        # AWR Loss
        # ============================================================================
        best_sample_idx = all_rewards.argmax(dim=0)
        
        target_trajectories = []
        target_weights = []

        for b in range(B):
            best_idx = best_sample_idx[b].item()
            best_traj = sampled_trajectories[best_idx][b:b+1]
            best_traj = best_traj.to(dtype=dtype) 
            
            advantage = advantages[best_idx, b].item()
            weight = torch.exp(torch.tensor(advantage / 2.0, device=device)).clamp(0.1, 5.0)

            target_trajectories.append(best_traj)
            target_weights.append(weight)

        target_trajectories = torch.cat(target_trajectories, dim=0)
        target_weights = torch.stack(target_weights).to(dtype=dtype)

        if target_trajectories.shape[-1] == 2:
            target_trajectories = F.pad(target_trajectories, (0, 1), value=0)

        target_future_rot = (
            torch.eye(3, device=device, dtype=dtype)
            .unsqueeze(0).unsqueeze(0)
            .expand(B, T_future, 3, 3)
            .contiguous()
        )

        target_action = self.action_space.traj_to_action(
            traj_history_xyz=ego_history_xyz.float(),
            traj_history_rot=ego_history_rot.float(),
            traj_future_xyz=target_trajectories.float(),
            traj_future_rot=target_future_rot.float(),
        ).to(dtype=dtype)

        # Forward through model
        t_train = torch.rand(B, device=device, dtype=dtype).view(B, 1, 1)
        x0_train = torch.randn(target_action.shape, device=device, dtype=dtype)
        x_t_train = (1 - t_train) * x0_train + t_train * target_action
        target_v_train = target_action - x0_train

        n_tokens = self.action_space.get_action_space_dims()[0]
        action_embeds_train = self.action_in_proj(x_t_train.to(dtype=dtype), t_train.to(dtype=dtype))
        if action_embeds_train.dim() == 2:
            action_embeds_train = action_embeds_train.view(B, n_tokens, -1)

        expert_out_train = self.expert(
            inputs_embeds=action_embeds_train,
            use_cache=False,
            return_dict=True,
        )
        last_hidden_train = expert_out_train.last_hidden_state
        pred_v_train = self.action_out_proj(last_hidden_train).view(
            B, *self.action_space.get_action_space_dims()
        )

        mse_per_sample = F.mse_loss(pred_v_train.float(), target_v_train.float(), reduction='none').mean(dim=[1, 2])
        rl_loss = (mse_per_sample * target_weights.float()).mean()

        # Compute metrics
        best_rewards = all_rewards.max(dim=0)[0]  # Best reward per batch item
        gt_is_best = (best_sample_idx == 0).float().mean().item()  # Fraction where GT is best

        metrics = {
            "rl_reward_mean": all_rewards.mean().item(),
            "rl_reward_best": best_rewards.mean().item(),
            "rl_gt_is_best": gt_is_best,
            "rl_safety_mean": np.mean(safety_scores_list),
            "rl_gt_sim_mean": np.mean(gt_similarity_list),
            "rl_reasoning_mean": np.mean(reasoning_alignment_list),
            "rl_advantage_std": advantages.std().item(),
            "rl_weight_mean": target_weights.mean().item(),
        }

        return rl_loss, metrics

    def _compute_trajectory_loss(self, gt_trajectory, ego_history_xyz, ego_history_rot):
        """Compute flow-matching loss in *action space* (same core logic as finetune_full.py).

        The repository's diffusion model is a sampler (BaseDiffusion.sample) and is not used as
        a direct forward module for training. Training predicts the vector field v(x_t, t).
        """
        B = gt_trajectory.shape[0]

        # Match expert device/dtype
        try:
            param = next(self.expert.parameters())
            device = param.device
            dtype = param.dtype
        except StopIteration:
            device = gt_trajectory.device
            dtype = torch.bfloat16

        gt_trajectory = gt_trajectory.to(device=device, dtype=dtype)
        ego_history_xyz = ego_history_xyz.to(device=device, dtype=dtype) if ego_history_xyz is not None else None
        ego_history_rot = ego_history_rot.to(device=device, dtype=dtype) if ego_history_rot is not None else None

        # Ensure trajectory has z=0 if only xy provided
        if gt_trajectory.shape[-1] == 2:
            gt_trajectory = F.pad(gt_trajectory, (0, 1), value=0)

        # Normalize ego history shapes
        # Accept (B, 16, 3) or (B, 1, 16, 3)
        if ego_history_xyz is None:
            raise ValueError("ego_history_xyz is required for trajectory loss")
        if ego_history_xyz.dim() == 4:
            ego_history_xyz = ego_history_xyz[:, 0]
        if ego_history_rot is not None and ego_history_rot.dim() == 5:
            ego_history_rot = ego_history_rot[:, 0]

        # If rotations missing, use identity
        if ego_history_rot is None or ego_history_rot.numel() == 0:
            T_hist = ego_history_xyz.shape[1]
            ego_history_rot = (
                torch.eye(3, device=device, dtype=dtype)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(B, T_hist, 3, 3)
                .contiguous()
            )

        # Convert future trajectory to action space (float32 for numerical stability)
        T_future = gt_trajectory.shape[1]
        gt_future_rot = (
            torch.eye(3, device=device, dtype=dtype)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(B, T_future, 3, 3)
            .clone()
        )

        try:
            gt_action = self.action_space.traj_to_action(
                traj_history_xyz=ego_history_xyz.float(),
                traj_history_rot=ego_history_rot.float(),
                traj_future_xyz=gt_trajectory.float(),
                traj_future_rot=gt_future_rot.float(),
            )
            gt_action = gt_action.to(dtype=dtype)
        except Exception as e:
            logger.warning(f"Failed to convert trajectory to action: {e}")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Flow matching: sample t ~ U(0,1), x0 ~ N(0,1), xt = (1-t)x0 + t x1
        t = torch.rand(B, device=device, dtype=dtype).view(B, 1, 1)
        x0 = torch.randn_like(gt_action)
        x_t = (1 - t) * x0 + t * gt_action
        target_v = gt_action - x0

        # Predict vector field with action_in_proj -> expert -> action_out_proj
        n_tokens = self.action_space.get_action_space_dims()[0]
        action_embeds = self.action_in_proj(x_t, t)
        if action_embeds.dim() == 2:
            action_embeds = action_embeds.view(B, n_tokens, -1)

        expert_out = self.expert(
            inputs_embeds=action_embeds,
            use_cache=False,
        )

        last_hidden = expert_out.last_hidden_state
        pred_v = self.action_out_proj(last_hidden).view(B, *self.action_space.get_action_space_dims())
        return F.mse_loss(pred_v, target_v)


# ==============================================================================
# Dataset (same as finetune_full.py)
# ==============================================================================

class FullPipelineDataset(torch.utils.data.Dataset):
    """Dataset for VLM + Trajectory training."""
    
    def __init__(self, data_path, processor, max_length=2048):
        self.data_path = data_path
        self.processor = processor
        self.max_length = max_length
        self.rl_image_size = (336, 336)  # Fixed size for RL batching

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

        if len(frames) == 0:
            raise RuntimeError(f"No valid frames found for sample idx={idx}. Checked: {frame_paths}")

        frames_tensor = torch.stack(frames, dim=0)

        # [NEW] Prepare single image for RL safety check (Resize to fixed size for batching)
        # Use the most recent frame (last one) for current view
        rl_img_pil = Image.open(frame_paths[-1]).convert("RGB").resize(self.rl_image_size)
        rl_image = torch.from_numpy(np.array(rl_img_pil)).permute(2, 0, 1).float() / 255.0  # (C, H, W)

        # Create message
        messages = helper.create_message(frames_tensor)

        # Add assistant response in the expected multimodal format
        reasoning = sample["reasoning"]
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": reasoning + "<|traj_future_start|>"}],
            }
        )
        
        # Tokenize
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
        except ValueError as e:
            # Common failure for Qwen3-VL when max_length is too small and truncation
            # cuts through expanded visual tokens.
            msg = str(e)
            if "Mismatch in `image` token count" in msg:
                raise ValueError(
                    f"apply_chat_template failed due to truncation with visual tokens. "
                    f"Increase --max_length (current: {self.max_length}) or reduce number of frames. "
                    f"Original error: {msg}"
                ) from e
            raise

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # Mask out the prompt portion (train only on assistant response)
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

        # Extract reasoning text for RL reward computation
        reasoning = sample.get("reasoning", "")

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "gt_trajectory": gt_traj,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "reasoning_text": reasoning,  # NEW: for RL
            "rl_image": rl_image,  # [NEW] Pass raw image for RL
        }

        # Match finetune_full.py: keep images flattened over the sample (num_images, ...)
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

        return output


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
    
    # Other fields
    for key in ["pixel_values", "image_grid_thw", "gt_trajectory", "ego_history_xyz", "ego_history_rot", "reasoning_text", "rl_image"]:
        if key not in batch[0] or batch[0][key] is None:
            continue
        vals = [item.get(key) for item in batch]
        if not all(v is not None for v in vals):
            continue

        # IMPORTANT: For Qwen3-VL, pixel_values and image_grid_thw are concatenated across the batch.
        if key in ["pixel_values", "image_grid_thw"]:
            result[key] = torch.cat(vals, dim=0)
        elif key == "reasoning_text":
            # Keep as list of strings
            result[key] = vals
        else:
            # rl_image, trajectories, etc. can be stacked
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

        # Ensure the process group is initialized with a clear rank<->GPU mapping.
        # This avoids NCCL warnings and reduces the chance of barrier hangs.
        if dist.is_available() and not dist.is_initialized():
            init_kwargs = {
                "backend": "nccl",
                "init_method": "env://",
                "world_size": world_size,
                "rank": rank,
            }
            try:
                if "device_id" in inspect.signature(dist.init_process_group).parameters:
                    init_kwargs["device_id"] = local_rank
            except Exception:
                pass
            dist.init_process_group(**init_kwargs)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Running on single device: {device}")

    def _dist_barrier():
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier(device_ids=[local_rank])
            except TypeError:
                dist.barrier()

    # Create output directory early so it's present even before the first checkpoint.
    if not getattr(training_args, "output_dir", None):
        raise ValueError("--output_dir is required")
    if rank == 0:
        os.makedirs(training_args.output_dir, exist_ok=True)
        print(f"Outputs will be saved under: {training_args.output_dir}")
    if world_size > 1:
        _dist_barrier()

    try:
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

        # Wrap in SRD-RL model
        model = ConsistencyEnhancedModel(
            base_model,
            traj_loss_weight=training_args.traj_loss_weight,
            consistency_loss_weight=training_args.consistency_loss_weight,
            temperature=training_args.consistency_temperature,
            # SRD-RL params
            safety_reward_weight=training_args.safety_reward_weight,
            gt_reward_weight=training_args.gt_reward_weight,
            reasoning_reward_weight=training_args.reasoning_reward_weight,
            num_trajectory_samples=training_args.num_trajectory_samples,
            rl_loss_weight=training_args.rl_loss_weight,
            danger_threshold=training_args.danger_keyword_threshold,
            gt_trust_min=training_args.gt_trust_min,
            gt_trust_max=training_args.gt_trust_max,
        )
        
        # Train consistency projections
        for param in model.vision_proj.parameters():
            param.requires_grad = True
        for param in model.lang_proj.parameters():
            param.requires_grad = True
        for param in model.traj_proj.parameters():
            param.requires_grad = True

        # Gradient checkpointing
        if training_args.gradient_checkpointing:
            model.vlm.enable_input_require_grads()

        # Wrap with DDP
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            if rank == 0:
                print("Model wrapped with DDP")

        # Dataset
        processor = helper.get_processor(base_model.tokenizer)
        dataset = FullPipelineDataset(
            data_path=data_args.data_path,
            processor=processor,
            max_length=training_args.max_length,
        )

        # DataLoader
        from torch.utils.data import DataLoader

        def custom_collate(batch):
            return collate_fn(batch, base_model.tokenizer)

        if world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
            dataloader = DataLoader(
                dataset,
                batch_size=training_args.per_device_train_batch_size,
                sampler=sampler,
                collate_fn=custom_collate,
                num_workers=4,
                pin_memory=training_args.dataloader_pin_memory,
            )
        else:
            sampler = None
            dataloader = DataLoader(
                dataset,
                batch_size=training_args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=custom_collate,
                num_workers=4,
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
            if world_size > 1 and sampler is not None:
                sampler.set_epoch(epoch)

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}") if rank == 0 else dataloader
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
                    reasoning_text=batch.get("reasoning_text"),  # NEW: for SRD-RL
                    rl_image=batch.get("rl_image"),  # [NEW] Pass RL image
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
                        # Format metrics for display
                        display_dict = {"loss": accumulated_loss * accumulation_steps}

                        # Group metrics for cleaner display
                        for k, v in loss_dict.items():
                            if k.startswith("rl_"):
                                # RL metrics with abbreviated names
                                short_name = k.replace("rl_", "").replace("_mean", "").replace("_std", "")
                                display_dict[short_name] = f"{v:.3f}"
                            else:
                                display_dict[k] = f"{v:.4f}"

                        pbar.set_postfix(display_dict)

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
                            torch.save(
                                {
                                    "expert_lora": model_to_save.expert.state_dict(),
                                    "action_in_proj": model_to_save.action_in_proj.state_dict(),
                                    "action_out_proj": model_to_save.action_out_proj.state_dict(),
                                    "step": global_step,
                                },
                                os.path.join(save_path, "expert_diffusion.pt"),
                            )
                            print(f"\nSaved full checkpoint at step {global_step}")
                        else:
                            print(f"\nSaved VLM LoRA at step {global_step}")

                    if world_size > 1:
                        _dist_barrier()

        # Save final
        if rank == 0:
            final_save_path = os.path.join(training_args.output_dir, "final")
            os.makedirs(final_save_path, exist_ok=True)

            model_to_save = model.module if isinstance(model, DDP) else model
            model_to_save.vlm.save_pretrained(os.path.join(final_save_path, "vlm_lora"))
            torch.save(
                {
                    "expert_lora": model_to_save.expert.state_dict(),
                    "action_in_proj": model_to_save.action_in_proj.state_dict(),
                    "action_out_proj": model_to_save.action_out_proj.state_dict(),
                },
                os.path.join(final_save_path, "expert_diffusion.pt"),
            )

            print("Training complete!")

        if world_size > 1:
            _dist_barrier()

    finally:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    train()
