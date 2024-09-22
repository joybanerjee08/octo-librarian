import streamlit as st
import requests
import faiss
import PyPDF2
import nltk
import docx
import json
import pickle
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

IBM_PROJECT_ID = os.getenv('IBM_PROJECT_ID')
IBM_API_KEY = os.getenv('IBM_API_KEY')

def get_api_token():
    url = "https://iam.cloud.ibm.com/identity/token"

    data = {
        "grant_type":"urn:ibm:params:oauth:grant-type:apikey",
        "apikey":IBM_API_KEY
    }

    headers = {
        "Content-Type":"application/x-www-form-urlencoded"
    }

    response = requests.post(url,data=data,headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data['access_token']
    else:
        return ""

def merge_dicts(dicts):
    """
    Merges multiple dictionaries into a single dictionary.

    Args:
        *dicts: A variable number of dictionaries to merge.

    Returns:
        dict: The merged dictionary.
    """

    merged_dict = {}
    i = 0
    for d in dicts:
        for k in d:
            merged_dict[i] = d[k]
            i+=1
    return merged_dict

def get_embeddings(token,text):
    headers = {
        'Authorization': 'Bearer '+token,
        'Accept': 'application/json',
        'Content-Type': 'application/json',
    }

    params = {
        'version': '2023-10-25',
    }

    data = '{\n  "inputs": '+json.dumps(text)+',\n  "model_id": "ibm/slate-30m-english-rtrvr-v2",\n  "project_id": "'+IBM_PROJECT_ID+'"\n}'

    response = requests.post('https://us-south.ml.cloud.ibm.com/ml/v1/text/embeddings', params=params, headers=headers, data=data)

    output = response.json()
    embeddings = output['results']
    output = {}
    for i in range(len(text)):
        output[i] = [text[i],embeddings[i]['embedding']]
    return output

def read_book_pdf(pdf_path, chunk_size=500):
    """
    Reads a PDF file, extracts its text, and splits it into chunks of a specified size.

    Args:
        pdf_path (str): The path to the PDF file.
        chunk_size (int): The desired size of each text chunk.

    Returns:
        list: A list of text chunks.
    """

    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""

        for page in reader.pages:
            text += page.extract_text()


        # Remove extra whitespace and normalize line breaks
        text = text.strip().replace('\n', ' ')

        # Split the text into chunks
        chunks = nltk.sent_tokenize(text)
        chunks = [chunk for chunk in chunks if chunk.strip()]  # Remove empty chunks
        chunks = [chunk for chunk in chunks if len(chunk) > 10]  # Filter out short chunks

        # Combine chunks to reach the desired chunk size
        combined_chunks = []
        current_chunk = ""
        for chunk in chunks:
            if len(current_chunk) + len(chunk) + 1 <= chunk_size:
                current_chunk += chunk + " "
            else:
                combined_chunks.append(current_chunk.strip())
                current_chunk = chunk + " "
        if current_chunk:
            combined_chunks.append(current_chunk.strip())

        return combined_chunks

def read_book_doc(word_file_path, chunk_size=500):
    """
    Reads a Word document (DOC or DOCX), extracts its text, and splits it into chunks of a specified size.

    Args:
        word_file_path (str): The path to the Word document.
        chunk_size (int): The desired size of each text chunk.

    Returns:
        list: A list of text chunks.
    """

    doc = docx.Document(word_file_path)
    text = ""

    for paragraph in doc.paragraphs:
        text += paragraph.text + " "

    # Remove extra whitespace and normalize line breaks
    text = text.strip().replace('\n', ' ')

    # Split the text into chunks
    chunks = nltk.sent_tokenize(text)
    chunks = [chunk for chunk in chunks if chunk.strip()]  # Remove empty chunks
    chunks = [chunk for chunk in chunks if len(chunk) > 10]  # Filter out short chunks

    # Combine chunks to reach the desired chunk size
    combined_chunks = []
    current_chunk = ""
    for chunk in chunks:
        if len(current_chunk) + len(chunk) + 1 <= chunk_size:
            current_chunk += chunk + " "
        else:
            combined_chunks.append(current_chunk.strip())
            current_chunk = chunk + " "
    if current_chunk:
        combined_chunks.append(current_chunk.strip())

    return combined_chunks

def read_book_txt(txt_file_path, chunk_size=500):
    """
    Reads a TXT file, extracts its text, and splits it into chunks of a specified size.

    Args:
        txt_file_path (str): The path to the TXT file.
        chunk_size (int): The desired size of each text chunk.

    Returns:
        list: A list of text chunks.
    """

    with open(txt_file_path, 'r') as txt_file:
        text = txt_file.read()

    # Remove extra whitespace and normalize line breaks
    text = text.strip().replace('\n', ' ')

    # Split the text into chunks
    chunks = nltk.sent_tokenize(text)
    chunks = [chunk for chunk in chunks if chunk.strip()]  # Remove empty chunks
    chunks = [chunk for chunk in chunks if len(chunk) > 10]  # Filter out short chunks

    # Combine chunks to reach the desired chunk size
    combined_chunks = []
    current_chunk = ""
    for chunk in chunks:
        if len(current_chunk) + len(chunk) + 1 <= chunk_size:
            current_chunk += chunk + " "
        else:
            combined_chunks.append(current_chunk.strip())
            current_chunk = chunk + " "
    if current_chunk:
        combined_chunks.append(current_chunk.strip())

    return combined_chunks

def store_embeddings(uploaded_file,token):
    if os.path.exists('emb_'+uploaded_file.name+'.pkl'):
        with open('emb_'+uploaded_file.name+'.pkl', 'rb') as handle:
            return pickle.load(handle)
    if 'pdf' in uploaded_file.name:
        chunk = read_book_pdf(uploaded_file.name)
    if 'doc' in uploaded_file.name or 'docx' in uploaded_file.name:
        chunk = read_book_doc(uploaded_file.name)
    if 'txt' in uploaded_file.name:
        chunk = read_book_doc(uploaded_file.name)
    with open('emb_'+uploaded_file.name+'.pkl', 'wb') as handle:
        embeddings = get_embeddings(token,chunk)
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return embeddings

def embedd_question(question,token):
    with open('emb_question.pkl', 'wb') as handle:
        embeddings = get_embeddings(token,[question])
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return embeddings

def find_context(question_embedding,embeddings):
    dim = 384
    index = faiss.IndexFlatL2(dim)
    for keys in embeddings:
        index.add(np.asarray(embeddings[keys][1]).reshape(1,-1))
    query_vector = np.asarray(question_embedding[0][1]).reshape(1,-1)
    k = 1
    distances, indices = index.search(query_vector.reshape(1, dim), k)
    closest_vector_index = indices[0][0]
    closest_vector = embeddings[closest_vector_index]
    return closest_vector[0]

def ask_granite(question,context,token):
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

    body = {
        "input": """<|context|>
    """+context+"""
    <|user|>
    Based on the context given, answer this question : """+question+"""
    <|assistant|>
    """,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 900,
            "repetition_penalty": 1.17
        },
        "model_id": "ibm/granite-13b-chat-v2",
        "project_id": IBM_PROJECT_ID
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer "+token
    }

    response = requests.post(
        url,
        headers=headers,
        json=body
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    return data['results'][0]['generated_text']

def main():
    nltk.download('punkt_tab')
    st.set_page_config(page_title="Octo Librarian", page_icon="octo-librarian.jpeg", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title("Octo Librarian")
    st.image("octo-librarian.jpeg",width=256)
    st.subheader("Your friendly librarian octopus, who reads all your given books to answer all your given questions !", divider="gray")
    st.image("IBM-Watson.jpg",width=128)
    st.subheader("Powered by IBM Watson", divider=True)

    uploaded_file1 = st.file_uploader("Choose document 1", type=["pdf", "doc", "docx", "txt"])

    file_uploaded = False

    if uploaded_file1 is not None:
        with open("./"+uploaded_file1.name, "wb") as f:
            f.write(uploaded_file1.getvalue())

        st.success("File 1 uploaded and saved successfully!")
        st.write(uploaded_file1.name)
        file_uploaded = True

    uploaded_file2 = st.file_uploader("Choose document 2", type=["pdf", "doc", "docx", "txt"])

    if uploaded_file2 is not None:
        with open("./"+uploaded_file2.name, "wb") as f:
            f.write(uploaded_file2.getvalue())

        st.success("File 2 uploaded and saved successfully!")
        st.write(uploaded_file2.name)
        file_uploaded = True

    uploaded_file3 = st.file_uploader("Choose document 3", type=["pdf", "doc", "docx", "txt"])

    if uploaded_file3 is not None:
        with open("./"+uploaded_file3.name, "wb") as f:
            f.write(uploaded_file3.getvalue())

        st.success("File 3 uploaded and saved successfully!")
        st.write(uploaded_file3.name)
        file_uploaded = True

    uploaded_file4 = st.file_uploader("Choose document 4", type=["pdf", "doc", "docx", "txt"])

    if uploaded_file4 is not None:
        with open("./"+uploaded_file4.name, "wb") as f:
            f.write(uploaded_file4.getvalue())

        st.success("File 4 uploaded and saved successfully!")
        st.write(uploaded_file4.name)
        file_uploaded = True

    uploaded_file5 = st.file_uploader("Choose document 5", type=["pdf", "doc", "docx", "txt"])

    if uploaded_file5 is not None:
        with open("./"+uploaded_file5.name, "wb") as f:
            f.write(uploaded_file5.getvalue())

        st.success("File 5 uploaded and saved successfully!")
        st.write(uploaded_file5.name)
        file_uploaded = True

    uploaded_file6 = st.file_uploader("Choose document 6", type=["pdf", "doc", "docx", "txt"])

    if uploaded_file6 is not None:
        with open("./"+uploaded_file6.name, "wb") as f:
            f.write(uploaded_file6.getvalue())

        st.success("File 6 uploaded and saved successfully!")
        st.write(uploaded_file6.name)
        file_uploaded = True

    uploaded_file7 = st.file_uploader("Choose document 7", type=["pdf", "doc", "docx", "txt"])

    if uploaded_file7 is not None:
        with open("./"+uploaded_file7.name, "wb") as f:
            f.write(uploaded_file7.getvalue())

        st.success("File 7 uploaded and saved successfully!")
        st.write(uploaded_file7.name)
        file_uploaded = True

    uploaded_file8 = st.file_uploader("Choose document 8", type=["pdf", "doc", "docx", "txt"])

    if uploaded_file8 is not None:
        with open("./"+uploaded_file8.name, "wb") as f:
            f.write(uploaded_file8.getvalue())

        st.success("File 8 uploaded and saved successfully!")
        st.write(uploaded_file8.name)
        file_uploaded = True

    if file_uploaded:
        question = st.text_input("Ask your question:")
        # Add the "Ask" button
        if st.button("Ask"):
            if question == "":
                st.info("You need to ask a question, to get an answer !")
            else:
                st.info("Reading your documents...")
                token = get_api_token()
                if token=="":
                    st.error("There was a problem with your token !")
                    return
                all_embeddings = []
                if uploaded_file1 is not None:
                    all_embeddings.append(store_embeddings(uploaded_file1,token))
                if uploaded_file2 is not None:
                    all_embeddings.append(store_embeddings(uploaded_file2,token))
                if uploaded_file3 is not None:
                    all_embeddings.append(store_embeddings(uploaded_file3,token))
                if uploaded_file4 is not None:
                    all_embeddings.append(store_embeddings(uploaded_file4,token))
                if uploaded_file5 is not None:
                    all_embeddings.append(store_embeddings(uploaded_file5,token))
                if uploaded_file6 is not None:
                    all_embeddings.append(store_embeddings(uploaded_file6,token))
                if uploaded_file7 is not None:
                    all_embeddings.append(store_embeddings(uploaded_file7,token))
                if uploaded_file8 is not None:
                    all_embeddings.append(store_embeddings(uploaded_file8,token))
                vector_question = embedd_question(question,token)
                context = find_context(vector_question,merge_dicts(all_embeddings))
                answer = ask_granite(question,context,token)
                st.balloons()
                st.success(answer)

if __name__ == "__main__":
    main()