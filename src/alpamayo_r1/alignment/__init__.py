# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alignment module for Alpamayo-R1 guideline-based self-improvement."""

from alpamayo_r1.alignment.guidelines import Guidelines, load_guidelines
from alpamayo_r1.alignment.openrouter_critic import OpenRouterCritic, CritiqueResult
from alpamayo_r1.alignment.preference_collector import PreferenceCollector
from alpamayo_r1.alignment.lora_trainer import LoRATrainer
from alpamayo_r1.alignment.alignment_loop import AlignmentLoop

__all__ = [
    "Guidelines",
    "load_guidelines",
    "OpenRouterCritic",
    "CritiqueResult",
    "PreferenceCollector",
    "LoRATrainer",
    "AlignmentLoop",
]
