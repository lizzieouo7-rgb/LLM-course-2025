"""
Assignment: Direct Preference Optimization (DPO) for LLM Alignment
Model: Qwen2.5-0.5B
Dataset: Intel/orca_dpo_pairs
Description: This script implements DPO using the PEFT/LoRA framework to align 
             a pre-trained language model with human preference data.
"""

import torch
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from huggingface_hub import login

# ==========================================
# 1. User Configuration
# ==========================================
# Replace with your credentials
HF_TOKEN = "" 
HF_USERNAME = "" 
NEW_MODEL_NAME = f"{HF_USERNAME}/Qwen2.5-0.5B-DPO-Aligned"

# ==========================================
# 2. Environment Setup
# ==========================================
print(">>> Initializing DPO Fine-tuning environment...")
login(token=HF_TOKEN)

# Determine the best available device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f">>> Computation backend: {device.upper()}")

# ==========================================
# 3. Dataset Processing
# ==========================================
print(">>> Loading preference dataset...")
# Using a subset for demonstration of the training pipeline
raw_dataset = load_dataset("Intel/orca_dpo_pairs", split="train[:1000]")

def chat_template_format(examples):
    """
    Format the raw dataset into the required DPO format: 
    'prompt', 'chosen', and 'rejected'.
    """
    new_examples = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    for system, question, chosen, rejected in zip(
        examples["system"], examples["question"], examples["chosen"], examples["rejected"]
    ):
        # Format prompt using standard ChatML-style headers
        prompt_text = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        new_examples["prompt"].append(prompt_text)
        new_examples["chosen"].append(chosen + "<|im_end|>")
        new_examples["rejected"].append(rejected + "<|im_end|>")
    return new_examples

processed_dataset = raw_dataset.map(chat_template_format, batched=True)
print(">>> Dataset processing complete.")

# ==========================================
# 4. Model & Tokenizer Initialization
# ==========================================
model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

print(f">>> Loading policy model: {model_id}")
# Load in half-precision for memory efficiency
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(device)

# PEFT/LoRA Configuration for DPO
# Note: DPO typically targets more modules for better alignment
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# ==========================================
# 5. DPO Training Configuration
# ==========================================
training_args = DPOConfig(
    output_dir="./dpo_results",
    beta=0.1,                          # KL-divergence penalty coefficient
    max_steps=100,                     # Total training iterations
    per_device_train_batch_size=1,     # Effective batch size management
    gradient_accumulation_steps=8,     
    learning_rate=5e-5,                # Optimized LR for DPO stability
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_steps=10,
    logging_steps=5,
    dataloader_num_workers=0,          # Multiprocessing disabled for compatibility
    remove_unused_columns=False,
    save_strategy="no",
    report_to="none"
)

# DPOTrainer handles both the policy model and implicit reference model
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,                    # PEFT handles the reference model automatically
    args=training_args,
    train_dataset=processed_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

# ==========================================
# 6. Execution & Hub Integration
# ==========================================
print(">>> Commencing DPO training...")
dpo_trainer.train()

print(">>> Saving fine-tuned adapters...")
dpo_trainer.model.save_pretrained("final_dpo_adapter")

print(f">>> Uploading model to Hugging Face Hub: {NEW_MODEL_NAME}")
try:
    dpo_trainer.model.push_to_hub(NEW_MODEL_NAME)
    tokenizer.push_to_hub(NEW_MODEL_NAME)
    print(">>> Alignment successfully completed and uploaded.")
except Exception as e:
    print(f">>> Upload error: {e}")

# Garbage collection to free resources
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None