# SahilBot 🤖

**SahilBot** is a machine learning chatbot built using Python and scikit-learn. It recognizes user intents and responds with emoji-rich, human-like replies for everyday conversations. Lightweight, customizable, and perfect for NLP learners.

---

## 📦 Features
- ML-powered intent recognition with scikit-learn
- Easy-to-edit `intents.json` file
- Fun, personality-driven replies with emojis
- Covers daily-life intents like greetings, jokes, help, and more

---

## 🛠️ Tech Stack
- Python 3.x
- scikit-learn (`LogisticRegression`)
- `CountVectorizer` (Bag-of-Words)
- JSON for training data
- Pickle for model persistence

---

## 🧠 How It Works
1. User input is vectorized using Bag-of-Words
2. Trained model predicts the intent tag
3. Bot replies with a random response mapped to that tag

---

## 🚀 Getting Started

### 🔧 Train the Bot
```bash
python train_bot.py
