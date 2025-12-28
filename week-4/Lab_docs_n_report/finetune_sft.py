"""
Task: Supervised Fine-tuning (SFT) of Qwen2.5-0.5B on Pirate Dataset
"""

import torch
import os
import shutil
import gc
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from huggingface_hub import login

HF_TOKEN = "" 
HF_USERNAME = "" 

# ==========================================
# 1. Setup & Authentication
# ==========================================
print(">>> Initializing project...")

# Construct model name
new_model_name = f"{HF_USERNAME}/Qwen2.5-0.5B-Pirate-Final"

# Login
login(token=HF_TOKEN)

# Clean up previous artifacts
if os.path.exists("final_adapter"):
    shutil.rmtree("final_adapter")

# Detect Hardware
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f">>> Device detected: {device.upper()}")

# ==========================================
# 2. Data Preparation
# ==========================================
model_name = "Qwen/Qwen2.5-0.5B"
dataset_name = "winglian/pirate-ultrachat-10k"

print(f">>> Loading dataset: {dataset_name}...")
dataset = load_dataset(dataset_name, split="train[:2000]")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def format_chat_template(row):
    row['text'] = tokenizer.apply_chat_template(
        row['messages'], 
        tokenize=False, 
        add_generation_prompt=False
    )
    return row

dataset = dataset.map(format_chat_template)

# ==========================================
# 3. Model Loading & LoRA Configuration
# ==========================================
print(f">>> Loading base model: {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map=None, 
    trust_remote_code=True
)
model.to(device)
model.config.use_cache = False 

peft_config = LoraConfig(
    r=16, 
    lora_alpha=16, 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM", 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ==========================================
# 4. Training Configuration
# ==========================================
training_args = SFTConfig(
    output_dir="./results",
    dataset_text_field="text",
    
    # Fast Run Settings (Modified for Demo)
    max_steps=60,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    logging_steps=1,
    
    # Mac Optimizations
    dataloader_num_workers=0,        
    dataloader_pin_memory=False,     
    optim="adamw_torch",             
    learning_rate=2e-4,
    fp16=False,                      
    save_strategy="no",              
    report_to="none",
    packing=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
    processing_class=tokenizer
)

# ==========================================
# 5. Execution
# ==========================================
print(">>> Starting training process...")
trainer.train()

# ==========================================
# 6. Visualization
# ==========================================
print(">>> Generating Loss Curve...")
log_history = trainer.state.log_history
steps = []
losses = []

for log in log_history:
    if "loss" in log and "step" in log:
        steps.append(log["step"])
        losses.append(log["loss"])

if steps and losses:
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label="Training Loss", color="blue", linewidth=2)
    plt.title("Model Training Loss Curve", fontsize=16)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig("training_loss.png")
    print("✅ Plot saved as 'training_loss.png'")

# ==========================================
# 7. Saving & Uploading
# ==========================================
print(">>> Saving adapter locally...")
trainer.model.save_pretrained("final_adapter")
tokenizer.save_pretrained("final_adapter")

print(">>> Cleaning memory for merging...")
del model, trainer
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(">>> Merging and Uploading to Hugging Face...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="cpu", 
    trust_remote_code=True
)
model_to_upload = PeftModel.from_pretrained(base_model, "final_adapter")
model_to_upload = model_to_upload.merge_and_unload()

try:
    model_to_upload.push_to_hub(new_model_name)
    tokenizer.push_to_hub(new_model_name)
    print(">>> ✅ SUCCESS: Model uploaded successfully!")
    print(f">>> URL: https://huggingface.co/{new_model_name}")
except Exception as e:
    print(f">>> ❌ Upload failed: {e}")