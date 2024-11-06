import os
import pickle
import streamlit as st
import time
import langchain
from langchain.llms import Ollama
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

st.title("News Research Tool")
st.sidebar.title("Add News Url")
url_input = []

for i in range(1):
    url = st.sidebar.text_input(f"Url {i +3} ")
    url_input.append(url)
button = st.sidebar.button("Process URLs")

file_path = "faiss_store.pkl"
placeholder = st.empty()
llm = Ollama(temperature = 0.9)
if button:
    loader = UnstructuredURLLoader(urls = url_input)
    placeholder.text("Data Loading...Started")
    data = loader.load()

    # split data
    placeholder.text("Text Splitter...Started")
    text_splitter = RecursiveCharacterTextSplitter(separators = ["\n\n", "\n", " "],
                                        chunk_size = 500, 
                                        chunk_overlap = 0,
                                        length_function = len  )
    chunks_docs = text_splitter.split_documents(data)
    print("Document length: ", len(chunks_docs))
    print("Chunk Documents: ", chunks_docs)
    placeholder.text("Text Splitted...")

    # create embeddings
    embeddings = OllamaEmbeddings(model = "llama2")
    placeholder.text("Embedding Vector Started Building...")
    vectordb = FAISS.from_documents(chunks_docs, embedding=embeddings)
    time.sleep(2)

    # save the Faiss index to pickle file 
    with open(file_path, "wb") as f:
        pickle.dump(vectordb, f)

query = placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_db = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever =vector_db.as_retriever())
            result = chain({"question": query})
            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources","")
            if sources:
                st.subheader("Sources: ")
                source_list = sources.split("\n")
                for source in source_list:
                    st.write(source)