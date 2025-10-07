from PIL import Image
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Qdrant
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader
import io,tempfile
from dotenv import load_dotenv
load_dotenv()

import os

os.environ["LANGCHAIN_TRACING"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"]= os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGSMITH_TRACING")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGSMITH_PROJECT")
os.environ["GROQ_API_KEY"] =os.getenv("GROQ_API_KEY")



def retriever_init(pdf_bytes):
    #process pdf
    # pdf_path='../files/gemma.pdf'
    # doc=fitz.open(pdf_path)
    # pdf_bytes = uploaded_file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_bytes)
        temp_pdf.flush()  # Make sure all bytes are written

        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(temp_pdf.name)
        pages = loader.load()

    embeddings=HuggingFaceEmbeddings(model="./local_models/all-MiniLM-L6-v2")
    chunker=SemanticChunker(embeddings=embeddings)
    chunks=chunker.split_documents(
        documents=pages
    )
 
    # client = QdrantClient(":memory:")
    vectorstore=Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        location=":memory:",
        collection_name="report"
    )
    
    dense_retriever=vectorstore.as_retriever(search_kwargs={'k':3})
    sparse_retriever=BM25Retriever.from_documents(chunks)
    sparse_retriever.k=3
    
    hybrid_retriever=EnsembleRetriever(
        retrievers=[dense_retriever,sparse_retriever],
        weights=[0.6,0.4]
    )

    return hybrid_retriever

    


    