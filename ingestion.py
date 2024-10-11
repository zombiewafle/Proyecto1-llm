import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from consts import INDEX_NAME

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

class Document:
    def __init__(self, text, metadata):
        self.page_content = text
        self.metadata = metadata

def parse_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    text = soup.get_text()
    return text

def ingest_docs(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            filepath = os.path.join(directory, filename)
            text = parse_html(filepath)
            document = Document(text, {'source': filepath})
            documents.append(document)
    print(f"Loaded {len(documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(documents)
    print(f"Prepared {len(documents)} documents for ingestion")

    PineconeVectorStore.from_documents(
        documents, embeddings, index_name=INDEX_NAME
    )
    print("Documents added to Pinecone successfully")

if __name__ == "__main__":
    ingest_docs("/home/zombiewafle/Descargas/skate_web_scrapping/")  # Use your actual path
