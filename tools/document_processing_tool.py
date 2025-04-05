from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

class DocumentProcessingTool:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()

    def process_documents(self, directory_path):
        # Load all PDF files from the directory
        pdf_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".pdf")]
        documents = []

        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(documents)

        # Store the documents in a FAISS vector store
        vector_store = FAISS.from_documents(split_documents, self.embeddings)
        return vector_store

    def compare_documents(self, vector_store, query):
        # Perform similarity search on the vector store
        results = vector_store.similarity_search(query, k=5)
        return [doc.page_content for doc in results]