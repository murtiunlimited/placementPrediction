🎓 Placement Prediction System
A machine learning-based web application that predicts student placement outcomes based on academic and demographic data. This project uses a Streamlit frontend and a Scikit-Learn backend.

The LLM used in this project was using a API Key from GROQ CLOUD.

Make your own API Key from GroqCloud and you can use the chatbot about 1000 times per month if on free plan (THIS MAY CHANGE)

📁 Project Structure
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
```
```text
🚀 How to Run Locally
Clone the Repository (Ensure you have the label encoder, model, and scaler files).

Open the folder in your terminal.

Install dependencies and navigate to the app directory:

pip install -r requirements.txt
cd app
Launch the application:

streamlit run app.py
Boom, done! You can now view the application in your local browser.
```

```text
📦 Requirements
If you prefer downloading libraries individually:
pandas
numpy
scikit-learn
streamlit
openai
chromadb
sentence-transformers
requests
```
```text
☁️ How to Run on Cloud (AWS EC2)
1. Instance Setup
Launch a new EC2 Ubuntu Machine on AWS.

Allocate at least 50 GB of storage.

Port Mapping: Ensure you open port 8501 in your AWS Security Group.

2. Install Docker
Connect to your Ubuntu instance via SSH and run the following commands:

sudo apt-get update -y  
sudo apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker

3. Build and Deploy
After installing Docker, build the image from the root folder:

docker build -t my-image:latest .
docker run -it -p 8501:8501 my-image:latest
4. Access the App
Open a new browser tab and navigate to: http://<YOUR-EC2-PUBLIC-IP>:8501
```
