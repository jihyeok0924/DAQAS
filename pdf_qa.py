import os
from PyPDF2 import PdfReader
from langchain import OpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langdetect import detect
import streamlit as st

os.environ["OPENAI_API_KEY"] = "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"

def process_file(uploaded_file):
    raw_text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text

    detected_language = detect(raw_text)
    question = st.text_input("Enter your question about the PDF:")

    if question:
        with st.spinner("Processing your question..."):
            model = ChatOpenAI(model="gpt-3.5-turbo")
            qa_chain = load_qa_chain(model, chain_type="map_reduce")
            qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
            answer = qa_document_chain.run(input_document=raw_text, question=question, language=detected_language)
        st.markdown(f"**Answer:**\n\n{answer}")

st.title("PDF QA Chatbot")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    process_file(uploaded_file)
