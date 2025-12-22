import os
import torch
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np
from PIL import Image

import transformers
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

"""
cd /home/byounggun/alpamayo/src
torchrun --nproc_per_node=4 -m alpamayo_r1.alignment.finetune_rellis3d \
    --data_path /home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset/finetune_data.jsonl \
    --output_dir /home/byounggun/alpamayo/outputs/alpamayo_lora_rellis3d \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --save_steps 100 \
    --bf16 True \
    --lora_r 8 \
    --lora_alpha 16
"""

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
    image_dir: str = field(default=None, metadata={"help": "Root directory for images (Rellis-3D)"})

@dataclass
class LoraTrainingArguments(transformers.TrainingArguments):
    """Training arguments with LoRA specific parameters."""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length."},
    )
    lora_r: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(
        default_factory=lambda: {"use_reentrant": False}
    )
    ddp_find_unused_parameters: bool = False

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

class SupervisedDataset(torch.utils.data.Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, processor, max_length: int = 2048):
        super(SupervisedDataset, self).__init__()
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
        
        # 1. Load Images
        # 'frame_paths' contains absolute paths or relative?
        # The generation script saved absolute paths in 'frame_paths'
        # But we need to make sure they exist.
        
        frames = []
        for fp in sample['frame_paths']:
            # Handle both absolute and relative paths if needed
            # Assuming absolute since generation script used Path objects
            try:
                img = Image.open(fp).convert("RGB")
                img_np = np.array(img)
                # Convert to tensor (C, H, W) and normalize handled by processor?
                # helper.create_message expects torch tensor normalized 0-1
                img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                frames.append(img_t)
            except Exception as e:
                logger.error(f"Error loading image {fp}: {e}")
                # Return dummy or fail? Fail for now
                raise e
                
        frames_tensor = torch.stack(frames, dim=0) # (4, C, H, W)
        
        # 2. Create Message (Prompt)
        # Using helper.create_message which formats standardized user prompt
        messages = helper.create_message(frames_tensor)
        
        # 3. Add Assistant Response (Target)
        # We append the reasoning as the assistant's response
        reasoning = sample['reasoning']
        
        # We need to construct the full conversation for the processor
        # helper.create_message returns a list of dictionaries
        
        # Add the reasoning to the last message or as a new assistant message?
        # Usually: User -> Assistant
        # messages already has the user part.
        
        messages.append({"role": "assistant", "content": [{"type": "text", "text": reasoning + "<|traj_future_start|>"}]})
        
        # 4. Tokenize
        # We use the processor to formatting
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False, # We added assistant message manualy
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        
        # extract from batch dimension
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        
        # 5. Labels
        # We need to mask the user prompt parts for training
        # This is a bit tricky with pre-tokenized inputs from apply_chat_template if it doesn't return offsets
        # For simplicity in this v1, we train on the whole sequence or try to find the split.
        # Ideally, we should use a DataCollatorForCompletionOnlyLM logic or similar.
        # But since reasoning is at the end, standard causal LM training on the whole sequence 
        # is often "okay" but less efficient. 
        # BETTER: Use the message structure to mask.
        # However, apply_chat_template does it all at once.
        
        # Strategy: Tokenize prompt only, find length. Then tokenize full.
        # Mask [0 : prompt_len].
        
        # Re-do tokenization safely
        # Prompt only
        prompt_messages = messages[:-1] # Remove assistant
        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True, # Add generation prompt for accurate length
            return_dict=True,
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]
        
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Mask prompt tokens
        
        # Build output dict with required VLM inputs
        output = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        # Include pixel_values and image_grid_thw for VLM forward pass
        # Note: processor returns (batch=1, ...) so we remove batch dim
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            if pv.dim() > 3:  # (batch, seq, channels) or similar
                pv = pv.squeeze(0)
            output["pixel_values"] = pv
        if "image_grid_thw" in inputs:
            igt = inputs["image_grid_thw"]
            # image_grid_thw should be (num_images, 3) after removing batch
            # Don't over-squeeze - keep at least 2D
            if igt.dim() == 3:  # (batch, num_images, 3)
                igt = igt.squeeze(0)  # -> (num_images, 3)
            elif igt.dim() == 2 and igt.shape[0] == 1:  # (1, 3) single image case
                pass  # Keep as (1, 3)
            output["image_grid_thw"] = igt
        return output

# Data Collator behaves differently for VLM usually
# we need to stack tensors and handle padding

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        # Handle extra keys if any (like images)
        # Note: AlpamayoR1's VLM likely expects 'pixel_values' or 'images' in forward
        # But apply_chat_template in 'test_rellis3d.py' usage:
        # **tokenized_data -> passed to model.generate
        # Checking test_rellis3d.py again...
        # inputs = processor.apply_chat_template(...)
        # model_inputs = {"tokenized_data": inputs, ...}
        # In AlpamayoR1.sample...: input_ids = tokenized_data.pop("input_ids")
        # .. generate( ..., **tokenized_data)
        # So 'inputs' from processor contains everything needed (images etc? No, usually separate)
        
        # WAIT. test_rellis3d.py passes 'frames_tensor' to helper.create_message.
        # helper.create_message puts images in 'content'.
        # processor.apply_chat_template CONSUMES these messages.
        # For Qwen-VL based models, it usually processes images and returns 'pixel_values'.
        
        # We need to collect other keys from instances if they exist
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        
        # We need to inspect what keys apply_chat_template returns in Dataset
        # Custom collator needed to collate pixel_values if present
        
        return batch

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, LoraTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load Model
    print(f"Loading Alpamayo model from {model_args.model_name_or_path}...")
    # Load full Alpamayo model, but we will train only .vlm
    # Use device_map="cpu" initially to avoid OOM, then move VLM to cuda
    # Or load regular and offload. 
    # Since we are using DDP/Trainer, we should be careful with manual device movement if Trainer handles it.
    # But here we are initializing before Trainer.
    
    full_model = AlpamayoR1.from_pretrained(
        model_args.model_name_or_path, 
        dtype=torch.bfloat16
    )
    # Do NOT move to CUDA yet. Let Trainer handle the VLM part.
    # We leave Expert and Diffusion on CPU (never moved to GPU).
    
    # MEMORY OPTIMIZATION: Ensure Expert and Diffusion stay on CPU
    # (They are already on CPU by default from_pretrained)
    
    model = full_model.vlm
    
    # Configure LoRA
    if training_args.lora_target_modules is None:
         training_args.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
         
    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=training_args.lora_target_modules,
        lora_dropout=training_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing if needed
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        
    model.print_trainable_parameters()

    # Create Dataset
    processor = helper.get_processor(full_model.tokenizer)
    dataset = SupervisedDataset(
        data_path=data_args.data_path,
        processor=processor,
        max_length=training_args.max_length
    )
    
    # Custom Collator that handles whatever the processor outputs
    def collate_fn(instances):
        # Stack inputs
        batch = {}
        
        # Keys to stack / pad
        keys = instances[0].keys()
        
        for k in keys:
            vals = [i[k] for i in instances]
            if k == "input_ids" or k == "labels":
                batch[k] = torch.nn.utils.rnn.pad_sequence(
                    vals, batch_first=True, padding_value=full_model.tokenizer.pad_token_id
                )
            elif k == "attention_mask":
                batch[k] = torch.nn.utils.rnn.pad_sequence(
                    vals, batch_first=True, padding_value=0
                )
            elif k == "pixel_values":
                # Qwen3-VL: concat all images along dim 0 (not stack!)
                batch[k] = torch.cat(vals, dim=0)
            elif k == "image_grid_thw":
                # Qwen3-VL: concat all image grid info along dim 0
                batch[k] = torch.cat(vals, dim=0)
            else:
                # Default stacking
                try:
                    batch[k] = torch.stack(vals)
                except:
                    batch[k] = vals
        return batch

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )
    
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer, training_args.output_dir)

if __name__ == "__main__":
    train()
