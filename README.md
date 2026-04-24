# 📱 Teen Phone Addiction — ANN Predictor

A **Streamlit web app** that uses an Artificial Neural Network (ANN) to predict teen phone addiction levels based on behavioral and lifestyle data.

🔗 **Live Demo:** *(Streamlit Cloud link yahan paste karo)*

---

## 📊 Features

- 🧠 **ANN Model** — TensorFlow/Keras regression model
- 🎛️ **Adjustable Hyperparameters** — epochs, batch size, dropout via sidebar
- 📈 **Live Plots** — Actual vs Predicted, Loss & MAE curves
- 🔮 **Live Predictor** — Custom input se real-time prediction
- 🔍 **Dataset Explorer** — Raw data aur stats table

---

## 🚀 Local Run

```bash
# 1. Clone karo
git clone https://github.com/YOUR_USERNAME/teen-phone-addiction-ann.git
cd teen-phone-addiction-ann

# 2. Dependencies install karo
pip install -r requirements.txt

# 3. App run karo
streamlit run app.py
```

---

## 📁 File Structure

```
├── app.py                     # Streamlit app
├── teen_phone_addiction.csv   # Dataset
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🧠 Model Architecture

```
Input Layer  →  Dense(64, ReLU)  →  Dropout
             →  Dense(32, ReLU)  →  Dropout
             →  Dense(16, ReLU)
             →  Dense(1)  [Regression Output]
```

**Target:** `Addiction_Level` (0–10 continuous scale)

---

## 📦 Tech Stack

- Python · TensorFlow · Scikit-learn · Pandas · Streamlit · Matplotlib
