import streamlit as st
import requests
import json


API_URL = "http://localhost:5000"  


def upload_pdf(file):
    files = {"file": file}
    response = requests.post(f"{API_URL}/upload", files=files)
    return response.json()

def query_llm(query):
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"query": query})
    response = requests.post(f"{API_URL}/query", headers=headers, data=data)
    return response.json()


st.set_page_config(page_title="PDF Uploader & Query LLM", layout="wide")
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a Mode", ["Upload PDF", "Query LLM"])


if app_mode == "Upload PDF":
    st.title("Upload PDF to Vector Store")
    st.markdown("Upload your PDF file below to store it for later querying.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Upload PDF"):
            with st.spinner("Processing..."):
                result = upload_pdf(uploaded_file)
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success(result['message'])

if app_mode == "Query LLM":
    st.title("Ask the LLM")
    st.markdown("Enter your query below to get an answer based on the uploaded PDF.")

    user_query = st.text_input("Your Query")

    if user_query:
        if st.button("Ask LLM"):
            with st.spinner("Fetching answer..."):
                result = query_llm(user_query)
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.write(f"Answer: {result['answer']}")
