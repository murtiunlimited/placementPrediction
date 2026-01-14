import os
import streamlit as st
import pandas as pd
import pickle
import requests
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, "dataset", "Placement_Data_Full_Class.csv")

GROQ_API_KEY = "gsk_XXXXXXXXXXXXXXXXXXX"
model = pickle.load(open("./app/placement_model.pkl", "rb"))
scaler = pickle.load(open("./app/scaler.pkl", "rb"))

st.set_page_config(page_title="Placement Predictor", layout="centered")
@st.cache_resource
def init_rag():
    df = pd.read_csv(dataset_path)

    documents = []
    for _, row in df.iterrows():
        documents.append(
            f"Gender: {row['gender']}, "
            f"SSC: {row['ssc_p']}%, "
            f"HSC: {row['hsc_p']}%, "
            f"Degree: {row['degree_p']}%, "
            f"MBA: {row['mba_p']}%, "
            f"Work Experience: {row['workex']}, "
            f"Specialisation: {row['specialisation']}, "
            f"Placement Status: {row['status']}"
        )

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    collection = client.get_or_create_collection("placements")

    if collection.count() == 0:
        collection.add(
            documents=documents,
            ids=[str(i) for i in range(len(documents))],
            embeddings=embed_model.encode(documents).tolist()
        )

    return collection, embed_model

collection, embed_model = init_rag()

def retrieve_context(query, k=3):
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )
    return results["documents"][0]

st.title("🎓 Campus Placement Prediction")
st.write("Enter student details and click **Predict Placement**")

st.divider()

gender = st.selectbox("Gender", ["Male", "Female"])
ssc_b = st.selectbox("SSC Board", ["Central", "Others"])
hsc_b = st.selectbox("HSC Board", ["Central", "Others"])
hsc_s = st.selectbox("HSC Stream", ["Commerce", "Science", "Arts"])
degree_t = st.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"])
workex = st.selectbox("Work Experience", ["Yes", "No"])
specialisation = st.selectbox("MBA Specialisation", ["Mkt&HR", "Mkt&Fin"])

st.subheader("Academic Scores")

ssc_p = st.number_input("SSC Percentage", 0.0, 100.0, 60.0)
hsc_p = st.number_input("HSC Percentage", 0.0, 100.0, 60.0)
degree_p = st.number_input("Degree Percentage", 0.0, 100.0, 60.0)
etest_p = st.number_input("E-Test Percentage", 0.0, 100.0, 60.0)
mba_p = st.number_input("MBA Percentage", 0.0, 100.0, 60.0)

gender_map = {"Male": 1, "Female": 0}
ssc_b_map = {"Central": 1, "Others": 0}
hsc_b_map = {"Central": 1, "Others": 0}
hsc_s_map = {"Commerce": 0, "Science": 1, "Arts": 2}
degree_t_map = {"Sci&Tech": 0, "Comm&Mgmt": 1, "Others": 2}
workex_map = {"Yes": 1, "No": 0}
specialisation_map = {"Mkt&HR": 0, "Mkt&Fin": 1}

if st.button("Predict Placement"):
    salary = 0

    input_df = pd.DataFrame([[
        gender_map[gender],
        ssc_b_map[ssc_b],
        hsc_b_map[hsc_b],
        hsc_s_map[hsc_s],
        degree_t_map[degree_t],
        workex_map[workex],
        specialisation_map[specialisation],
        ssc_p, hsc_p, degree_p, etest_p, mba_p,
        salary
    ]], columns=model.feature_names_in_)

    num_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("Placed ")
        st.session_state.prediction = "Placed"
    else:
        st.error("Not Placed ")
        st.session_state.prediction = "Not Placed"

st.divider()
st.subheader("Career Guidance Chatbot (RAG)")

### RAG BOT
user_query = st.chat_input("Ask about placements, scores, or work experience")

if user_query:
    retrieved_docs = retrieve_context(user_query)

    context = "\n".join(retrieved_docs)

    prompt = f"""
You are a campus placement advisor.
Answer ONLY using the context below.

Context:
{context}

Question:
{user_query}
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a helpful placement expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
    )

    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]

        with st.chat_message("assistant"):
            st.write(answer)

        with st.expander("📚 Retrieved examples from dataset"):
            for doc in retrieved_docs:
                st.write("-", doc)
    else:
        st.error("Are you dumb????")
