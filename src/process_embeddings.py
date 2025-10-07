
from transformers import CLIPProcessor,CLIPModel
from PIL import Image
import base64
import torch
import fitz #pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
import io
from langchain_community.retrievers import BM25Retriever
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings
import uuid
import numpy as np
from langchain.retrievers import EnsembleRetriever


from dotenv import load_dotenv
load_dotenv()

import os

os.environ["LANGCHAIN_TRACING"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"]= os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGSMITH_TRACING")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGSMITH_PROJECT")
os.environ["GROQ_API_KEY"] =os.getenv("GROQ_API_KEY")




#clip model for both text and image embeddings
clip_model=CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_processor=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')




def embed_image(image_data):
    if isinstance(image_data,str): #path
        image=Image.open(image_data).convert('RGB')
    else:
        image=image_data

    inputs=clip_processor(images=image,return_tensors='pt')
    with torch.no_grad():
        features=clip_model.get_image_features(**inputs)
        #normalize to unit vector
        features=features/features.norm(dim=-1,keepdim=True)
        return features.squeeze().numpy()
    
def embed_text(text):
    inputs=clip_processor(
        text=text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=77
    )
    with torch.no_grad():
        features=clip_model.get_text_features(**inputs)
        #normalize to unit vector
        features=features/features.norm(dim=-1,keepdim=True)
        return features.squeeze().numpy()
    
def process_embeddings_from_pdf(doc):
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    all_docs=[]
    all_embeddings=[]
    image_data_store={}

    for i,page in enumerate(doc):
            #process text 
            text=page.get_text()
            if text.strip():
                temp_doc=Document(page_content=text,metadata={'page':i,'type':'text'})
                text_chunks=splitter.split_documents([temp_doc])
                for chunk in text_chunks:
                    embedding=embed_text(chunk.page_content)
                    all_embeddings.append(embedding)
                    all_docs.append(chunk)
    for i,page in enumerate(doc):
        #process text 
        text=page.get_text()
        if text.strip():
            temp_doc=Document(page_content=text,metadata={'page':i,'type':'text'})
            text_chunks=splitter.split_documents([temp_doc])
            for chunk in text_chunks:
                embedding=embed_text(chunk.page_content)
                all_embeddings.append(embedding)
                all_docs.append(chunk)

    #process images
    #convert pdf images to pil
    #convert to base64
    #clip embedding for retrieval
    for img_id,img in enumerate(page.get_images(full=True)):
        try:
            xref=img[0]
            base_image=doc.extract_image(xref)
            image_bytes=base_image['image']

            #convert to pil
            pil_image=Image.open(io.BytesIO(image_bytes)).convert('RGB')

            #image identifier
            image_id=f'page_{i}_img_{img_id}'

            buffered=io.BytesIO()
            pil_image.save(buffered,format="PNG")
            image_base64=base64.b64encode(buffered.getvalue()).decode()
            image_data_store[image_id]=image_base64

            #embed using clip
            embedding=embed_image(pil_image)
            all_embeddings.append(embedding)

            #document for image
            image_doc=Document(
                page_content=f'[image:{image_id}]',
                metadata={'page':i,'type':'image','image_id':image_id}
            )
            all_docs.append(image_doc)

        except Exception as e:
            print(e,img_id)
            continue
    doc.close()
    return all_docs,all_embeddings,image_data_store

def store_embeddings_qdrant(all_embeddings,all_docs):
    class CLIPEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [embed_text(t) for t in texts]
        def embed_query(self, text):
            return embed_text(text)

    clip_embeddings = CLIPEmbeddings()

    client = QdrantClient(":memory:")
    vector_dim = len(all_embeddings[0])

    # Create collection 
    collection_name='report'
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist() if isinstance(emb, np.ndarray) else emb,
            payload={
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            },
        )
        for doc, emb in zip(all_docs, all_embeddings)
    ]

    client.upsert(collection_name=collection_name, points=points)


    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=clip_embeddings,  # since embeddings are precomputed
    )
    return vectorstore


def create_multimodel_message(query,retrieved_docs,doc,image_data_store):
    content=[]
    content.append({
        'type':'text',
        'text':f'Question:{query}\n\nContext:\n'
    })
    text_docs=[doc for doc in retrieved_docs if doc.metadata.get('type')=='text']
    image_docs=[doc for doc in retrieved_docs if doc.metadata.get('type')=='image']

    if text_docs:
        text_context='\n\n'.join([
            f'[page {doc.metadata['page']}]: {doc.page_content}'
            for doc in text_docs
        ])
        content.append({
            'type':'text',
            'text':f'text excerpts:\n{text_context}\n'
        })

    #add images
    if image_docs:
        image_id=doc.metadata.get('image_id')
        if image_id and image_id in image_data_store:
            content.append({
                'type':'image',
                'text':f'\n[Image from page {doc.metadata['page']}]:\n'
            })
            content.append({
                'type':'image_url',
                'image_url':{
                    'url':f'data:image/png;base64,{image_data_store[image_id]}'
                }
            })  
    content.append({
        'type':'text',
        'text':'\nPlease answer the question based on the provided text and images'
    })

    return HumanMessage(content=content)

def retriever_init(pdf_bytes):
    #process pdf
    # pdf_path='../files/gemma.pdf'
    # doc=fitz.open(pdf_path)
    # pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    all_docs,all_embeddings,image_data_store=process_embeddings_from_pdf(doc)
    vectorstore=store_embeddings_qdrant(all_embeddings,all_docs)
    
    dense_retriever=vectorstore.as_retriever(search_kwargs={'k':5})
    sparse_retriever=BM25Retriever.from_documents(all_docs)
    sparse_retriever.k=3
    
    hybrid_retriever=EnsembleRetriever(
        retrievers=[dense_retriever,sparse_retriever],
        weights=[0.6,0.4]
    )

    

    return hybrid_retriever

    


    