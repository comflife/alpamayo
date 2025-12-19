#!/usr/bin/env python3
"""Test script for alignment module."""

import sys
sys.path.insert(0, '/home/byounggun/alpamayo/src')

print("Testing alignment module...")

# Test 1: Guidelines
print("\n=== Test 1: Guidelines ===")
try:
    from alpamayo_r1.alignment.guidelines import load_guidelines, Guidelines
    guidelines = load_guidelines()
    print(f"✓ Guidelines loaded: {len(guidelines)} rules")
    print(f"✓ Enabled rules: {len(guidelines.get_enabled_rules())}")
    print(f"\nPrompt:\n{guidelines.to_prompt()}")
except Exception as e:
    print(f"✗ Guidelines failed: {e}")

# Test 2: OpenRouter Critic (without API call)
print("\n=== Test 2: OpenRouter Critic (import only) ===")
try:
    from alpamayo_r1.alignment.openrouter_critic import OpenRouterCritic, CritiqueResult
    print("✓ OpenRouterCritic imported")
    # Don't instantiate without API key
except Exception as e:
    print(f"✗ OpenRouterCritic failed: {e}")

# Test 3: Preference Collector
print("\n=== Test 3: Preference Collector ===")
try:
    from alpamayo_r1.alignment.preference_collector import PreferenceCollector
    collector = PreferenceCollector(save_dir="/tmp/test_preferences")
    print(f"✓ PreferenceCollector created, size: {collector.size}")
except Exception as e:
    print(f"✗ PreferenceCollector failed: {e}")

# Test 4: LoRA Trainer (import only)
print("\n=== Test 4: LoRA Trainer (import only) ===")
try:
    from alpamayo_r1.alignment.lora_trainer import LoRATrainer, LoRAConfig
    print("✓ LoRATrainer imported")
except Exception as e:
    print(f"✗ LoRATrainer failed: {e}")

# Test 5: Alignment Loop (import only)
print("\n=== Test 5: Alignment Loop (import only) ===")
try:
    from alpamayo_r1.alignment.alignment_loop import AlignmentLoop
    print("✓ AlignmentLoop imported")
except Exception as e:
    print(f"✗ AlignmentLoop failed: {e}")

print("\n=== All tests complete ===")
