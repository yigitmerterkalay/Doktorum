import os
import json
import sys
import faulthandler
from datasets import Dataset

# Transformers importlarını geciktiriyoruz (eğer importta patlıyorsa yakalayalım diye)
# from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer

faulthandler.enable()

def log(msg):
    print(msg, flush=True)

BASE_DIR = r"C:/Users/yigit/OneDrive/Desktop/doktorum_deneme-main/Doktorum"
DATA_FILE = os.path.join(BASE_DIR, "Q_A-Data.js")
OUTPUT_DIR = os.path.join(BASE_DIR, "t5_model_trained")
LOG_DIR = os.path.join(BASE_DIR, "logs")

log("1) Script başladı.")
log(f"2) DATA_FILE = {DATA_FILE}")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Dosya yok: {DATA_FILE}")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

log(f"3) JSON yüklendi. Kayıt sayısı: {len(data)}")

dataset = Dataset.from_dict({
    "input": [item["input"] for item in data],
    "output": [item["output"] for item in data]
})
log(f"4) Dataset oluşturuldu: {dataset.num_rows} satır")

dataset = dataset.train_test_split(test_size=0.2, seed=42)
log(f"5) Split tamam: train={dataset['train'].num_rows} test={dataset['test'].num_rows}")

log("6) Model + tokenizer yükleniyor...")

try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
    log("6.1) transformers import OK")

    # Not: İnternet/HF indirme sorunu varsa burada patlar ve hata yazdırılır.
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    log("6.2) tokenizer OK")

    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    log("7) model OK (yükleme tamam)")

except Exception as e:
    log("!! Model/tokenizer yüklemede hata yakaladım:")
    log(repr(e))
    # stacktrace’i zorla bas
    import traceback
    traceback.print_exc()
    sys.exit(1)

def preprocess_function(examples):
    inputs = [f"semptom analizi: {text}" for text in examples["input"]]
    targets = examples["output"]

    model_inputs = tokenizer(
        inputs,
        max_length=256,
        truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        targets,
        max_length=256,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

log("8) Tokenize başlıyor (map)...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)
log("9) Tokenize bitti.")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    logging_dir=LOG_DIR,
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

log("10) Eğitim başlıyor...")
train_result = trainer.train()
log("11) Eğitim bitti.")
log(str(train_result))

log("12) Model kaydediliyor...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
log(f"13) Model kaydedildi: {OUTPUT_DIR}")

log("99) Dosyanın sonuna geldim.")
