# 🎓 Placement Prediction System

A machine learning-based web application that predicts student placement outcomes based on academic and demographic data. This project uses a **Streamlit** frontend and a **Scikit-Learn** backend.

## 📁 Project Structure

```text
placementPrediction/
├── app/
│   ├── app.py                   # Streamlit web application
│   ├── placement_model.pkl      # Trained ML model
│   ├── scaler.pkl               # Data scaler for preprocessing
│── dataset/
│   └── Placement_Data_Full_Class.csv
├── cleaning.py                  # Script for data cleaning & feature engineering
├── modeltrain.py                # Script for model training and serialization
├── requirements.txt             # Project dependencies
├── Dockerfile                   # (Coming Soon)
└── README.md
