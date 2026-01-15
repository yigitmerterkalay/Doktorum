# Doktorum

**Doktorum** is a desktop-based health assistant application that integrates a locally trained language model with a simple graphical user interface to analyze user-entered symptoms and return textual feedback.

The project is designed as a technical proof-of-concept to demonstrate local model training, controlled prompting, and real application integration without relying on external AI APIs.

---

## Features

- User registration and authentication (SQLite + bcrypt)
- Local storage of basic health data (pulse, blood pressure, blood sugar)
- Symptom analysis using:
  - Rule-based responses for critical keywords
  - A locally trained transformer model (DistilGPT-2)
- Lightweight Tkinter-based GUI
- Fully offline inference (no external API dependency)

---

## Project Structure

- `Doktorum.py` – Main application and GUI
- `distilgpt2_trained/` – Trained model files (excluded from repository)
- `distilgpt2_trained.py` – Model training script
- `test_local_model.py` – Local model testing script
- `Q_A-Data.js` – Training dataset
