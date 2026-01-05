
# 🎓 Campus Placement AI Chatbot

**(Streamlit · Machine Learning · RAG · Ollama)**

This project is a **Campus Placement Prediction and Career Guidance Chatbot** built using **Streamlit**, **Machine Learning**, and a **local Large Language Model (LLM)** powered by **Ollama**.

The chatbot uses **Retrieval-Augmented Generation (RAG)** with **ChromaDB** to answer career and placement-related questions grounded in **real student placement data**.

The system:

* Predicts whether a student is likely to be **Placed / Not Placed**
* Provides **data-driven career guidance** based on the placement dataset

---

## 🚀 How to Run Locally

1. Install the required dependencies
2. Navigate to the app directory
3. Pull the required Ollama models
4. Start Ollama and run the Streamlit app

```bash
pip install -r requirements.txt
cd app
ollama pull llama3.2
ollama pull nomic-embed-text
ollama serve
streamlit run app.py
```

---

## 📂 Project Structure

```text
project-root/
├── app/
│   ├── apps.py
│   ├── placement_model.pkl
│   ├── scaler.pkl
│   └── dataset/
│       └── Placement_Data_Full_Class.csv
├── requirements.txt
└── README.md
```

---

## 🤖 Features

* 🎯 Campus placement prediction (**Placed / Not Placed**)
* 💬 Career guidance chatbot using **llama3.2**
* 📚 RAG pipeline using **ChromaDB** and real placement data
* 🔒 Fully local setup (no external APIs required)
* 🖥 Interactive **Streamlit UI**

---

## 🧪 Example Queries

* *Do students without work experience get placed?*
* *What MBA percentage is common among placed students?*
* *Does specialization affect placement chances?*

---

## 🛠 Requirements

* Python **3.9+**
* Streamlit
* Ollama
* ChromaDB
* Standard data science libraries (NumPy, Pandas, Scikit-learn, etc.)

---
