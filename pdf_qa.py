import os
from PyPDF2 import PdfReader
from langchain import OpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langdetect import detect

os.environ["OPENAI_API_KEY"] = "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"

def process_file(file_path):
    with open(file_path, 'rb') as uploaded_file:
        raw_text = ""

        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text

        detected_language = detect(raw_text)

        question = input("Enter your question about the PDF: ")
        model = ChatOpenAI(model="gpt-3.5-turbo")
        qa_chain = load_qa_chain(model, chain_type="map_reduce")
        qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
        answer = qa_document_chain.run(input_document=raw_text, question=question, language=detected_language)
        print("Answer:", answer)

if __name__ == '__main__':
    file_path = input("Enter the path of your PDF file: ")
    process_file(file_path)
