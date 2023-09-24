import os
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain import OpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langdetect import detect
from langchain.agents import create_pandas_dataframe_agent
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
import pandas as pd
import tempfile
import sys
import os
from langchain.prompts import PromptTemplate

# pysqlite3 패키지의 라이브러리 경로 설정
pysqlite3_path = '/home/jihyeok/.local/lib/python3.9/site-packages/pysqlite3'
sys.path.insert(0, pysqlite3_path)
import pysqlite3

# API키를 환경변수로 설정
os.environ["OPENAI_API_KEY"] = "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" 

# 텍스트를 임시 파일로 저장하는 함수
def save_text_to_temp_file(raw_text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as temp_file:
        temp_file.write(raw_text)
        return temp_file.name

# 경로 내의 임시 파일을 삭제하는 함수
def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            pass

# Chroma 데이터 베이스를 생성하는 함수
def create_chroma_db(raw_text):
    persist_directory = '/home/jihyeok/바탕화면/database'
    clear_directory(persist_directory) 
    temp_file_path = save_text_to_temp_file(raw_text)
    text_loader = TextLoader(file_path=temp_file_path)
    document = text_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(document)
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=split_documents, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()
    os.remove(temp_file_path)
    return vectordb

# 파일 경로를 추출하는 함수
def get_file_extension(file_path):
    return os.path.splitext(file_path)[1].lower()[1:]
    
# 파일들을 처리하는 함수
def process_file(uploaded_file, file_type):
    raw_text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp", mode="wb") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    if file_type == 'pdf':
        reader = PdfReader(temp_file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text
    elif file_type == 'docx':
        doc = Document(temp_file_path)
        for p in doc.paragraphs:
            raw_text += p.text

    detected_language = None

    if file_type in ['pdf', 'docx']:
        vectordb = create_chroma_db(raw_text)

        template = """
        You are an AI chatbot that generates answers based on the uploaded document.
        Please provide accurate and specific answers based on what can be found in the document.

        1. Be specific and provide details in your question for a more accurate answer.
        2. Please enter your question about the document or provide details for a specific answer.
        3. If you have a general question about the document, please ask.
        4. If you want a mathematical problem solved, please specify the problem and include any relevant details.

        Please answer as kindly as you can!
        """
        prompt = PromptTemplate.from_template(template) 

        question = st.text_input("Enter your question about the document: ")
        prompted_text = prompt.format() 
        combined_question = prompted_text + " " + question
        if st.button("Process"):
            retriever = vectordb.as_retriever(search_kwargs={"k": 6})
            docs = retriever.get_relevant_documents(combined_question)
            relevant_docs_content = "\\n".join([doc.page_content for doc in docs])
            model = ChatOpenAI(model="gpt-3.5-turbo")
            qa_chain = load_qa_chain(model, chain_type="map_reduce")
            qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
            answer = qa_document_chain.run(input_document=relevant_docs_content, question=combined_question, language=detected_language)
            st.write("Answer:", answer)

    elif file_type == 'csv':
            question = st.text_input("Enter your question about the CSV data: ")
            if st.button("Process"):
                df = pd.read_csv(temp_file_path)
                agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
                answer = agent.run(question)
                st.write("Answer:", answer)

# Streamlit을 실행
if __name__ == '__main__':
    st.title("Document Auto Q&A System")
    file_path = st.file_uploader("Upload a file", type=["pdf", "docx", "csv"])

    if file_path:
        file_type = get_file_extension(file_path.name)
        if file_type in ["pdf", "docx", "csv"]:
            process_file(file_path, file_type)
        else:
            st.error("Unsupported file type. Please upload a PDF, DOCX, or CSV file.")
