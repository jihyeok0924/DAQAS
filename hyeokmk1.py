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
    clear_directory(persist_directory) # 데이터 베이스 초기화
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

# 파일들을 처리하는 함수수
def process_file(uploaded_file, file_type):
    raw_text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp", mode="wb") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Extracting text from the file based on its type
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

    detected_language = None  # CSV 파일의 경우 언어 감지를 수행하지 않도록 None으로 설정

    if file_type in ['pdf', 'do행
if __name__ == '__main__':
    st.title("Document Auto Q&A System")
    file_path = st.file_uploader("Upload a file", type=["pdf", "docx", "csv"])

    if file_path:
        file_type = get_file_extension(file_path.name)
        if file_type in ["pdf", "docx", "csv"]:
            process_file(file_path, file_type)
        else:
            st.error("Unsupported file type. Please upload a PDF, DOCX, or CSV file.")
