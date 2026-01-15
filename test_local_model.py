import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = r"C:/Users/yigit/OneDrive/Desktop/doktorum_deneme-main/Doktorum"
MODEL_DIR = os.path.join(BASE_DIR, "distilgpt2_trained")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

# CPU için
model.eval()

def generate(symptom: str):
    prompt = f"Semptom: {symptom}\nCevap:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    while True:
        s = input("Semptom gir (çıkmak için q): ").strip()
        if s.lower() == "q":
            break
        print("\n--- MODEL ---")
        print(generate(s))
        print("------------\n")
