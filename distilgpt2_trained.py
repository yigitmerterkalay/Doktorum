import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# =========================
# 0) PATHS (senin projene göre)
# =========================
BASE_DIR = r"C:/Users/yigit/OneDrive/Desktop/doktorum_deneme-main/Doktorum"
DATA_FILE = os.path.join(BASE_DIR, "Q_A-Data.js")
OUTPUT_DIR = os.path.join(BASE_DIR, "distilgpt2_trained")

print("1) Script başladı.")
print("2) DATA_FILE =", DATA_FILE)
print("3) OUTPUT_DIR =", OUTPUT_DIR)

# =========================
# 1) JSON yükle
# =========================
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Dosya yok: {DATA_FILE}")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list) or len(data) == 0:
    raise ValueError("JSON boş veya beklenen formatta değil (liste olmalı).")

print(f"4) JSON yüklendi. Kayıt sayısı: {len(data)}")

# Beklenen alanlar: input, output
for i, item in enumerate(data[:5]):
    if "input" not in item or "output" not in item:
        raise ValueError("JSON içinde 'input' ve 'output' alanları olmalı.")

# =========================
# 2) Dataset oluştur
#    (Soru + Cevap tek metne dönüyor)
# =========================
def to_text(ex):
    # Basit şablon: modelin öğrenmesi için sabit format
    inp = str(ex["input"]).strip()
    out = str(ex["output"]).strip()
    return f"Semptom: {inp}\nCevap: {out}\n"

texts = [to_text(item) for item in data]
dataset = Dataset.from_dict({"text": texts})
dataset = dataset.train_test_split(test_size=0.2, seed=42)

print(f"5) Dataset hazır. train={len(dataset['train'])} test={len(dataset['test'])}")

# =========================
# 3) Tokenizer + Model (DistilGPT-2)
# =========================
MODEL_NAME = "distilgpt2"
print("6) Model + tokenizer yükleniyor:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# GPT2 tokenizer'da pad token yok, ekliyoruz
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# =========================
# 4) Tokenize
# =========================
def preprocess(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )

tokenized = dataset.map(preprocess, batched=True, remove_columns=["text"])

# Dil modeli collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# =========================
# 5) Training args
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=25,

    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=5e-5,
    weight_decay=0.01,

    fp16=False,  # CPU ise False kalmalı
    report_to="none",
)

# =========================
# 6) Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# =========================
# 7) Train + Save
# =========================
print("7) Eğitim başlıyor...")
trainer.train()
print("8) Eğitim bitti.")

print("9) Model kaydediliyor...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("10) Kaydedildi:", OUTPUT_DIR)

print("11) Tamam.")
