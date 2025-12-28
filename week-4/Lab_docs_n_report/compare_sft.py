"""
Course: LLM Fine-tuning Project
Task: Final Performance Comparison (Base vs. Fine-tuned)
Hardware: Apple Silicon (Mac M4)
Features: Stable Inference with Repetition Penalty
Author: Lizzie Su
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

# ==========================================
# 1. Configuration
# ==========================================
base_model_name = "Qwen/Qwen2.5-0.5B"
adapter_path = "./final_adapter" 
device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

def generate_stable_response(model, prompt, label):
    """Generates response using stable beam search to avoid empty/looping output"""
    print(f"\n{'='*25} {label} {'='*25}")
    
    # Using a clearer prompt to guide the model
    messages = [
        {"role": "system", "content": "You are a helpful pirate assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Streamer to see the response word by word
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    print(f"User Query: {prompt}")
    print(f"Response: ", end="", flush=True)
    
    with torch.no_grad():
        model.generate(
            **inputs, 
            max_new_tokens=50,          # Sufficient for short conversation
            do_sample=False,           # Disable sampling for maximum stability
            repetition_penalty=1.5,    # Strongly prevent "aye aye aye" loops
            no_repeat_ngram_size=3,    # Prevent repeating 3-word phrases
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer
        )

# ==========================================
# 2. Comparison Execution
# ==========================================
print(f">>> Initializing comparison on {device}...")

# Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    torch_dtype=torch.float16, 
    device_map=None,
    trust_remote_code=True
).to(device)

test_query = "Hello, how are you today?"

# Test 1: Original Model
generate_stable_response(base_model, test_query, "ORIGINAL QWEN MODEL")

# Test 2: Fine-tuned Model (Load Adapter)
print("\n>>> Attaching Pirate Adapter...")
pirate_model = PeftModel.from_pretrained(base_model, adapter_path)
pirate_model.to(device)

generate_stable_response(pirate_model, test_query, "FINE-TUNED (PIRATE) MODEL")

print("\n" + "="*60)
print("Observation: The fine-tuned model should now consistently start with 'Ahoy' or 'Matey'.")