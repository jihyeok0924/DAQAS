import os
from PyPDF2 import PdfReader
from docx import Document
from langchain import OpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langdetect import detect
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd

os.environ["OPENAI_API_KEY"] = "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"

def process_file(file_path, file_type):
    with open(file_path, 'rb') as uploaded_file:
        raw_text = ""

        if file_type == 'pdf':
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text
        elif file_type == 'docx':
            doc = Document(uploaded_file)
            for p in doc.paragraphs:
                raw_text += p.text
        elif file_type == 'csv':
            df = pd.read_csv(uploaded_file)
            raw_text = df.to_json(orient='split')

        detected_language = detect(raw_text)

        if file_type in ['pdf', 'docx']:
            llm = OpenAI(temperature=0)
            summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
            summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
            summary = summarize_document_chain.run(input_document=raw_text, language=detected_language)
            print("Summary:", summary)

            # Answering questions for PDF and DOCX files
            question = input("Enter your question about the document: ")
            model = ChatOpenAI(model="gpt-3.5-turbo")
            qa_chain = load_qa_chain(model, chain_type="map_reduce")
            qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
            answer = qa_document_chain.run(input_document=raw_text, question=question, language=detected_language)
            print("Answer:", answer)
        elif file_type == 'csv':
            question = input("Enter your question about the CSV data: ")
            df = pd.read_json(raw_text, orient='split')
            agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
            answer = agent.run(question)
            print("Answer:", answer)

if __name__ == '__main__':
    file_path = input("Enter the path of your file: ")
    file_type = input("Enter the file type (pdf, docx, csv): ")
    process_file(file_path, file_type)
