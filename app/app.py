import streamlit as st
import pandas as pd
import numpy as np
import pickle
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://host.docker.internal:11434/v1"
API_KEY = "ollama"

model = pickle.load(open("app/placement_model.pkl", "rb"))
scaler = pickle.load(open("app/scaler.pkl", "rb"))


def create_embedding_function():
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=API_KEY,
        api_base=OLLAMA_BASE_URL,
        model_name=EMBED_MODEL,
    )

def create_llm_client():
    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=API_KEY,
    )

def generate_completion(client, messages):
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content

def load_placement_csv():
    df = pd.read_csv("app/dataset/Placement_Data_Full_Class.csv")

    documents = []
    for _, row in df.iterrows():
        text = (
            f"Gender: {row['gender']}, "
            f"SSC: {row['ssc_p']}%, "
            f"HSC: {row['hsc_p']}%, "
            f"Degree: {row['degree_p']}%, "
            f"MBA: {row['mba_p']}%, "
            f"Work Experience: {row['workex']}, "
            f"Specialisation: {row['specialisation']}, "
            f"Placement Status: {row['status']}"
        )
        documents.append(text)

    return documents

def setup_chromadb(documents, embedding_fn):
    client = chromadb.Client()

    try:
        client.delete_collection("placements")
    except:
        pass

    collection = client.create_collection(
        name="placements",
        embedding_function=embedding_fn,
    )

    collection.add(
        documents=documents,
        ids=[str(i) for i in range(len(documents))],
    )

    return collection

def find_related_chunks(query, collection, top_k=3):
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )
    return results["documents"][0]

def rag_pipeline(query, collection, llm_client):
    related_chunks = find_related_chunks(query, collection)
    context = "\n".join(related_chunks)

    prompt = f"""
You are a campus placement advisor.
Answer ONLY using the context below.

Context:
{context}

Question:
{query}
"""

    response = generate_completion(
        llm_client,
        [
            {"role": "system", "content": "You are a helpful placement expert."},
            {"role": "user", "content": prompt},
        ]
    )

    return response, related_chunks

@st.cache_resource
def init_rag():
    documents = load_placement_csv()
    embedding_fn = create_embedding_function()
    llm_client = create_llm_client()
    collection = setup_chromadb(documents, embedding_fn)
    return collection, llm_client

collection, llm_client = init_rag()

st.title("Campus Placement Prediction")

st.write("---")

# =====================================
# INPUT FIELDS
# =====================================
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    ssc_b = st.selectbox("SSC Board", ["Central", "Others"])
    hsc_b = st.selectbox("HSC Board", ["Central", "Others"])
    workex = st.selectbox("Work Experience", ["Yes", "No"])

with col2:
    hsc_s = st.selectbox("HSC Stream", ["Commerce", "Science", "Arts"])
    degree_t = st.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"])
    specialisation = st.selectbox("MBA Specialisation", ["Mkt&HR", "Mkt&Fin"])

with st.expander("📊 Academic Scores"):
    ssc_p = st.slider("SSC %", 0, 100, 60)
    hsc_p = st.slider("HSC %", 0, 100, 60)
    degree_p = st.slider("Degree %", 0, 100, 60)
    etest_p = st.slider("E-Test %", 0, 100, 60)
    mba_p = st.slider("MBA %", 0, 100, 60)

# =====================================
# ENCODING
# =====================================
gender_map = {"Male": 1, "Female": 0}
ssc_b_map = {"Central": 1, "Others": 0}
hsc_b_map = {"Central": 1, "Others": 0}
hsc_s_map = {"Commerce": 0, "Science": 1, "Arts": 2}
degree_t_map = {"Sci&Tech": 0, "Comm&Mgmt": 1, "Others": 2}
workex_map = {"Yes": 1, "No": 0}
specialisation_map = {"Mkt&HR": 0, "Mkt&Fin": 1}

# =====================================
# PREDICTION
# =====================================
if st.button("🔮 Predict Placement", use_container_width=True):
    feature_order = model.feature_names_in_
    salary = 0

    input_df = pd.DataFrame(
        [[
            gender_map[gender],
            ssc_b_map[ssc_b],
            hsc_b_map[hsc_b],
            hsc_s_map[hsc_s],
            degree_t_map[degree_t],
            workex_map[workex],
            specialisation_map[specialisation],
            ssc_p, hsc_p, degree_p, etest_p, mba_p, salary
        ]],
        columns=feature_order
    )

    num_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("🎉 Placement Status: Placed")
    else:
        st.error("❌ Placement Status: Not Placed")

st.write("---")

# =====================================
# RAG CHATBOT
# =====================================
st.subheader("🤖 Career Guidance Chatbot (RAG)")
user_input = st.text_input("Ask about placements, scores, work experience:")

if st.button("Chat"):
    if user_input.strip():
        answer, sources = rag_pipeline(
            user_input,
            collection,
            llm_client
        )

        st.markdown(f"**Chatbot 🤖:** {answer}")

        with st.expander("📚 Retrieved examples from dataset"):
            for s in sources:
                st.write("-", s)
