"""
Task: DPO Performance Comparison (Base Model vs. DPO Aligned Model)
Description: Evaluating instruction following and reasoning quality.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

# ==========================================
# 1. Configuration
# ==========================================
base_model_name = "Qwen/Qwen2.5-0.5B"
# Ensure this matches the folder name where your DPO adapter was saved
dpo_adapter_path = "./final_dpo_adapter" 
device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

def generate_dpo_response(model, prompt, label):
    """Stable inference for DPO comparison"""
    print(f"\n{'='*25} {label} {'='*25}")
    
    # Using a standard instruction-following system prompt
    messages = [
        {"role": "system", "content": "You are a helpful and logical assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    print(f"User Query: {prompt}")
    print(f"Response: ", end="", flush=True)
    
    with torch.no_grad():
        model.generate(
            **inputs, 
            max_new_tokens=150,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.2, # Essential for 0.5B models
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer
        )

# ==========================================
# 2. Execution
# ==========================================
print(f">>> Comparing models on {device}...")

# Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    torch_dtype=torch.float16, 
    device_map=None,
    trust_remote_code=True
).to(device)

# --- Test Case: Logical Reasoning ---
test_query = "Which is larger, 0.9 or 0.11? Explain your reasoning step-by-step."

# Test Original Model
generate_dpo_response(base_model, test_query, "ORIGINAL QWEN MODEL")

# Load DPO Adapter
print(f"\n>>> Loading DPO Adapter from {dpo_adapter_path}...")
dpo_model = PeftModel.from_pretrained(base_model, dpo_adapter_path)
dpo_model.to(device)

# Test DPO Model
generate_dpo_response(dpo_model, test_query, "DPO ALIGNED MODEL")

print("\n" + "="*60)