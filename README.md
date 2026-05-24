# 🚨 Smart Incident Severity Prediction System
---

## 🧠 Project Summary

An end-to-end AI-powered traffic incident severity prediction system
that combines Machine Learning + Natural Language Processing to predict accident severity levels (0–3) 
using structured features (weather, time, location, road conditions) and unstructured text data (incident descriptions). 
The system is designed to support **emergency response decision-making** by providing fast, accurate, and data-driven severity classification.

---

## ⚙️ Core Pipeline

Data Cleaning → Feature Engineering → Outlier Handling → Label Encoding
→TF-IDF Text Vectorization → Feature Fusion → Scaling → XGBoost Model Training 
→ Class Imbalance Handling → Hyperparameter Tuning → Evaluation → Model Serialization

---

## 🚀 Key Highlights

- 🧠 XGBoost classifier (Best performing model)
- 📝 NLP integration using TF-IDF (1–2 grams)
- ⚖️ Class imbalance handling (class weights)
- 🔍 Hyperparameter tuning (RandomizedSearchCV)
- 📊 Advanced feature engineering (time, weather, spatial)
- 📈 Model evaluation using F1-score, confusion matrix
- 💾 Saved artifacts: model, scaler, vectorizer, features

---

## 📊 Model Performance

| Model | Accuracy | F1 Score |
|------|----------|----------|
| XGBoost | ~85% | ~0.85 |
| Random Forest | ~73% | ~0.72 |
| Decision Tree | ~69% | ~0.68 |
| Logistic Regression | ~55% | ~0.54 |

---

## 🧠 Final Model Insights

The final optimized **XGBoost model** achieved the best performance due to its ability to:
- Capture nonlinear relationships in accident data
- Handle mixed structured + sparse NLP features
- Reduce overfitting using regularization
- Improve minority class detection using weighting strategy

---

## 📌 Input Features

- Geographic: Latitude, Longitude  
- Environmental: Weather, Temperature, Humidity, Visibility, Wind Speed  
- Temporal: Hour, Day, Month  
- Road conditions: Traffic signals, junctions, crossings  
- Text: Incident description (processed using NLP)

---

## 🎯 Output

Predicts accident severity levels:
- 0 → Low Risk  
- 1 → Medium Risk  
- 2 → High Risk  
- 3 → Critical Risk  

---

## 🧪 Evaluation Metrics

- Accuracy Score  
- Macro / Weighted F1-Score  
- Confusion Matrix Analysis  
- Feature Importance Ranking  

---

## 🧱 System Architecture

Modular ML pipeline with separation between:
- Preprocessing Layer  
- Feature Engineering Layer  
- NLP Processing Layer  
- Model Training Layer  
- Inference Layer  
- Deployment Layer (Streamlit-ready)

---

## 🛠️ Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, NLTK, TF-IDF, SciPy, Matplotlib, Seaborn, Joblib, Streamlit-ready pipeline

---

## 💾 Saved Artifacts

- `xgb_accident_model.pkl`
- `scaler.pkl`
- `tfidf_vectorizer.pkl`
- `features.pkl`
- `best_threshold.pkl`

---

## 👨‍💻 Author

**Ahmed Faried Almasry**  
📧 Email: a7medalma3ry@gmail.com  
🐙 GitHub: https://github.com/a7med-16 

---

## ⭐ Call To Action

If you find this project useful:⭐ Star the repository  

📩 Contact for collaboration or improvement ideas  

---

## 🚀 Impact

This system demonstrates how **Machine Learning + NLP
can be applied to real-world safety systems to improve emergency response speed, reduce uncertainty, and support smarter traffic management decisions.
