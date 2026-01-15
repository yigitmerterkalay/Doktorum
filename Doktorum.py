import os
import threading
import tkinter as tk
from tkinter import messagebox
import sqlite3
import bcrypt
import json
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# LOCAL MODEL AYARLARI
# =========================
MODEL_DIR = r"C:/Users/yigit/OneDrive/Desktop/doktorum_deneme-main/Doktorum/distilgpt2_trained"


class LocalSymptomModel:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model klasörü bulunamadı: {model_dir}")

        # Tokenizer + Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # GPT-2 türevlerinde pad_token yoksa generate bazen saçmalar/patlar
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, symptom_text: str) -> str:
        symptom_text = (symptom_text or "").strip()
        if not symptom_text:
            return "Semptom girmedin. Ben de kristal küre taşımıyorum."

        # Burada modeli “istediğin gibi konuşturma” kısmı var:
        # Prompt formatını sabit ve disiplinli tutarsan saçmalama azalır.
        prompt = (
            "Sen bir sağlık asistanısın.\n"
            "Türkçe, kısa, net ve maddeli cevap ver.\n"
            "Tanı koyma. Sadece olasılıklar, evde yapılabilecekler ve acil uyarılar.\n"
            "Cevap 6-8 maddeyi geçmesin.\n\n"
            f"Semptom: {symptom_text}\n"
            "Cevap:\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Promptu kesip sadece cevap kısmını göster
        if "Cevap:" in text:
            text = text.split("Cevap:", 1)[-1].strip()

        return text.strip()[:1200]


# =========================
# VERİTABANI
# =========================
conn = sqlite3.connect("health_app.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password BLOB NOT NULL,
    age INTEGER,
    gender TEXT,
    height REAL,
    weight REAL,
    health_data TEXT
)
""")
conn.commit()

current_user = None


# =========================
# AUTH
# =========================
def register_user(username, password, age, gender, height, weight):
    try:
        hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        cursor.execute(
            "INSERT INTO users (username, password, age, gender, height, weight, health_data) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (username, hashed_password, age, gender, height, weight, ""),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def login_user(username, password):
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    if result:
        stored_password = result[0]
        # sqlite bazen bytes yerine memoryview döndürebilir
        if isinstance(stored_password, memoryview):
            stored_password = stored_password.tobytes()
        if bcrypt.checkpw(password.encode("utf-8"), stored_password):
            return True
    return False


# =========================
# RULE-BASED
# =========================
def rule_based_response(symptoms: str):
    rules = {
        "öksürük": "Öksürük için bol sıvı tüketin. 1 haftadan uzun sürerse veya nefes darlığı eşlik ederse sağlık kuruluşuna başvurun.",
        "göğüs ağrısı": "Göğüs ağrısı ciddi olabilir. Özellikle baskı/yanma, nefes darlığı, terleme, kola/çeneye vurma varsa acile başvurun.",
        "ateş": "Ateş varsa sıvı alımını artırın, dinlenin. Yüksek ateş 3 günden uzun sürerse veya kötüleşirse sağlık uzmanına danışın.",
    }
    for keyword, response in rules.items():
        if keyword in symptoms:
            return response
    return None


# =========================
# UI
# =========================
root = tk.Tk()
root.title("Sağlık Uygulaması")
root.geometry("420x330")

# Local modeli uygulama açılırken bir kere yükle
try:
    local_model = LocalSymptomModel(MODEL_DIR)
except Exception as e:
    local_model = None
    messagebox.showerror("Model Hatası", f"Local model yüklenemedi:\n{e}")


def show_menu(username):
    global current_user
    current_user = username

    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text=f"Hoş geldiniz, {username}!", font=("Helvetica", 16)).pack(pady=10)

    tk.Button(root, text="Sağlık Verisi Ekle", command=add_health_data).pack(fill=tk.X, padx=10, pady=5)
    tk.Button(root, text="Sağlık Verilerini Görüntüle", command=view_health_data).pack(fill=tk.X, padx=10, pady=5)
    tk.Button(root, text="Semptom Analizi Yap", command=analyze_symptoms).pack(fill=tk.X, padx=10, pady=5)
    tk.Button(root, text="Çıkış Yap", command=root.quit).pack(fill=tk.X, padx=10, pady=5)


def analyze_symptoms():
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="Semptom Analizi", font=("Helvetica", 16)).pack(pady=10)
    tk.Label(root, text="Semptomlarınızı girin:").pack(pady=5)

    symptom_entry = tk.Entry(root, width=40)
    symptom_entry.pack(pady=5)

    # sonuçları pencerede de gösterelim (messagebox yerine daha düzgün)
    result_text = tk.Text(root, width=50, height=8, wrap="word")
    result_text.pack(padx=10, pady=10)

    def analyze():
        symptoms = symptom_entry.get().lower().strip()
        if not symptoms:
            messagebox.showwarning("Uyarı", "Semptom gir.")
            return

        analyze_btn.config(state="disabled")
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "Analiz ediliyor... (CPU ise biraz sürebilir)\n")

        # önce kural tabanı
        rb = rule_based_response(symptoms)
        if rb:
            result_text.delete("1.0", tk.END)
            result_text.insert(tk.END, rb)
            analyze_btn.config(state="normal")
            return

        # local model yoksa patlatmadan söyle
        if local_model is None:
            result_text.delete("1.0", tk.END)
            result_text.insert(tk.END, "Local model yüklenemedi. MODEL_DIR yolunu ve bağımlılıkları kontrol et.")
            analyze_btn.config(state="normal")
            return

        # UI donmasın diye thread
        def worker():
            try:
                response = local_model.generate(symptoms)
            except Exception as e:
                response = f"Hata:\n{e}"

            def update_ui():
                result_text.delete("1.0", tk.END)
                result_text.insert(tk.END, response)
                analyze_btn.config(state="normal")

            root.after(0, update_ui)

        threading.Thread(target=worker, daemon=True).start()

    analyze_btn = tk.Button(root, text="Analiz Et", command=analyze)
    analyze_btn.pack(pady=5)

    tk.Button(root, text="Geri Dön", command=lambda: show_menu(current_user)).pack(pady=5)


def add_health_data():
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="Sağlık Verisi Ekle", font=("Helvetica", 16)).pack(pady=10)

    current_date = datetime.now().strftime("%Y-%m-%d")
    tk.Label(root, text=f"Tarih: {current_date}", font=("Helvetica", 10), anchor="e").pack(pady=5, anchor="e")

    tk.Label(root, text="Nabız:").pack(pady=5)
    pulse_entry = tk.Entry(root)
    pulse_entry.pack(pady=5)

    tk.Label(root, text="Tansiyon (Büyük):").pack(pady=5)
    systolic_entry = tk.Entry(root)
    systolic_entry.pack(pady=5)

    tk.Label(root, text="Tansiyon (Küçük):").pack(pady=5)
    diastolic_entry = tk.Entry(root)
    diastolic_entry.pack(pady=5)

    tk.Label(root, text="Şeker:").pack(pady=5)
    sugar_entry = tk.Entry(root)
    sugar_entry.pack(pady=5)

    def save_health_data():
        pulse = pulse_entry.get()
        systolic = systolic_entry.get()
        diastolic = diastolic_entry.get()
        sugar = sugar_entry.get()

        if pulse and systolic and diastolic and sugar:
            health_data = json.dumps({
                "pulse": pulse,
                "systolic": systolic,
                "diastolic": diastolic,
                "sugar": sugar,
                "date": current_date
            }, ensure_ascii=False)

            cursor.execute("UPDATE users SET health_data = ? WHERE username = ?", (health_data, current_user))
            conn.commit()
            messagebox.showinfo("Başarılı", "Sağlık verileri kaydedildi.")
            show_menu(current_user)
        else:
            messagebox.showerror("Hata", "Lütfen tüm alanları doldurun.")

    tk.Button(root, text="Kaydet", command=save_health_data).pack(pady=10)
    tk.Button(root, text="Geri Dön", command=lambda: show_menu(current_user)).pack(pady=5)


def view_health_data():
    cursor.execute("SELECT health_data FROM users WHERE username = ?", (current_user,))
    result = cursor.fetchone()

    if result and result[0]:
        health_data = json.loads(result[0])
        for widget in root.winfo_children():
            widget.destroy()

        tk.Label(root, text="Sağlık Verileri", font=("Helvetica", 16)).pack(pady=10)

        health_info = (
            f"Tarih: {health_data.get('date','-')}\n"
            f"Nabız: {health_data.get('pulse','-')}\n"
            f"Tansiyon (Büyük): {health_data.get('systolic','-')}\n"
            f"Tansiyon (Küçük): {health_data.get('diastolic','-')}\n"
            f"Şeker: {health_data.get('sugar','-')}"
        )
        tk.Label(root, text=health_info, justify=tk.LEFT, font=("Helvetica", 12)).pack(pady=10)

        tk.Button(root, text="Geri Dön", command=lambda: show_menu(current_user)).pack(pady=5)
    else:
        messagebox.showinfo("Sağlık Verileri", "Kayıtlı sağlık verisi bulunamadı.")


def show_registration_screen():
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="Kayıt Ol", font=("Helvetica", 16)).pack(pady=10)

    tk.Label(root, text="Kullanıcı Adı:").pack(pady=5)
    username_entry = tk.Entry(root)
    username_entry.pack(pady=5)

    tk.Label(root, text="Şifre:").pack(pady=5)
    password_entry = tk.Entry(root, show="*")
    password_entry.pack(pady=5)

    tk.Label(root, text="Yaş:").pack(pady=5)
    age_entry = tk.Entry(root)
    age_entry.pack(pady=5)

    tk.Label(root, text="Cinsiyet (E/K):").pack(pady=5)
    gender_entry = tk.Entry(root)
    gender_entry.pack(pady=5)

    tk.Label(root, text="Boy (cm):").pack(pady=5)
    height_entry = tk.Entry(root)
    height_entry.pack(pady=5)

    tk.Label(root, text="Kilo (kg):").pack(pady=5)
    weight_entry = tk.Entry(root)
    weight_entry.pack(pady=5)

    def save_registration():
        username = username_entry.get()
        password = password_entry.get()
        age = age_entry.get()
        gender = gender_entry.get()
        height = height_entry.get()
        weight = weight_entry.get()

        if username and password and age and gender and height and weight:
            try:
                age_i = int(age)
                height_f = float(height)
                weight_f = float(weight)

                if register_user(username, password, age_i, gender, height_f, weight_f):
                    messagebox.showinfo("Başarılı", "Kayıt başarıyla oluşturuldu!")
                    show_login_screen()
                else:
                    messagebox.showerror("Hata", "Bu kullanıcı adı zaten kullanılıyor.")
            except ValueError:
                messagebox.showerror("Hata", "Lütfen tüm alanları doğru formatta girin.")
        else:
            messagebox.showerror("Hata", "Lütfen tüm alanları doldurun.")

    tk.Button(root, text="Kaydet", command=save_registration).pack(pady=10)
    tk.Button(root, text="Geri Dön", command=show_login_screen).pack(pady=5)


def show_login_screen():
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="Giriş Yap", font=("Helvetica", 16)).pack(pady=10)

    tk.Label(root, text="Kullanıcı Adı:").pack(pady=5)
    username_entry = tk.Entry(root)
    username_entry.pack(pady=5)

    tk.Label(root, text="Şifre:").pack(pady=5)
    password_entry = tk.Entry(root, show="*")
    password_entry.pack(pady=5)

    def login():
        username = username_entry.get()
        password = password_entry.get()

        if login_user(username, password):
            show_menu(username)
        else:
            messagebox.showerror("Hata", "Kullanıcı adı veya şifre yanlış.")

    tk.Button(root, text="Giriş Yap", command=login).pack(pady=5)
    tk.Button(root, text="Kayıt Ol", command=show_registration_screen).pack(pady=5)


# Uygulama başlangıcı
show_login_screen()
root.mainloop()
