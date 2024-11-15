from PyPDF2 import PdfReader
from dotenv import load_dotenv
import streamlit as st 
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models.ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

def get_pdfs(pdf_files):
    text = ""
    if pdf_files:
        for pdf in pdf_files:
            pdf_reader = PdfReader(pdf)
            for pages in pdf_reader.pages:
                text = pages.extract_text()
    return text

def get_chunks(pdf):
    text_splitter = RecursiveCharacterTextSplitter(
        separators= ["\n","\n\n", "."],
        chunk_size= 1000,
        chunk_overlap = 200
    )
    chunks = text_splitter.split_text(pdf)
    return chunks

def vectorStore(chunks):
    embeddings = OllamaEmbeddings(
            model = "llama2"
        )
    db = FAISS.from_texts(chunks, embedding=embeddings)
    db.save_local("faiss_index")
    return db

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model  = ChatOllama(model = "llama2", temperature = 0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt = prompt)
    return chain

def user_handle(question):
   embeddings = OllamaEmbeddings(
            model = "llama2"
        )
   
   load_faiss = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
   docs = load_faiss.similarity_search(question)
   chain = get_conversational_chain()
   response =  chain(
        {"input_documents":docs, "question": question}
        , return_only_outputs=True)
   st.write(response["output_text"])

def main():
    load_dotenv()
    st.set_page_config("AI Assitant")
    st.subheader("Upload your pdf")
    pdf_file =st.file_uploader("", type="pdf" ,accept_multiple_files=True)
    query = st.text_input("Ask a Question from the PDF Files")

    pdf = get_pdfs(pdf_file)
    chunks = get_chunks(pdf)
    print(len(chunks))
    if chunks:
        # st.text("Creating Vector Embedding")
        vectorStore(chunks)

    if query:
        user_handle(query)

if __name__ == '__main__':
    main()
