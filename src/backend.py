from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

## Reducers
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages:Annotated[list,add_messages]

from dotenv import load_dotenv
load_dotenv()

import os

os.environ["LANGCHAIN_TRACING"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"]= os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGSMITH_TRACING")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGSMITH_PROJECT")
os.environ["GROQ_API_KEY"] =os.getenv("GROQ_API_KEY")
os.environ["OPENWEATHERMAP_API_KEY"]=os.getenv("OPENWEATHERMAP_API_KEY")

from langchain_huggingface import HuggingFaceEmbeddings
from transformers import CLIPProcessor,CLIPModel
from PIL import Image
import base64

clip_model=CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_processor=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
# clip_model.eval()

#embedding function for images
import torch

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
    
def invoke_agent(uploaded_file,query):
    #process pdf
    import fitz #pymupdf
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.messages import HumanMessage,AIMessage
    from langchain_core.prompts import ChatMessagePromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    # pdf_path='../files/gemma.pdf'
    # doc=fitz.open(pdf_path)
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")


    all_docs=[]
    all_embeddings=[]
    image_data_store={}

    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    from langchain_core.documents import Document
    import io

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

    from langchain_community.retrievers import BM25Retriever
    sparse_retriever=BM25Retriever.from_documents(all_docs)
    sparse_retriever.k=3


    from qdrant_client import QdrantClient
    from qdrant_client.http.models import VectorParams, Distance, PointStruct
    from langchain_qdrant import QdrantVectorStore
    from langchain_core.embeddings import Embeddings
    import uuid
    import numpy as np

    class CLIPEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [embed_text(t) for t in texts]
        def embed_query(self, text):
            return embed_text(text)

    clip_embeddings = CLIPEmbeddings()

    # -----------------------------
    # 1️⃣ Create in-memory Qdrant
    # -----------------------------
    client = QdrantClient(":memory:")

    collection_name = "report"

    # Infer embedding size
    vector_dim = len(all_embeddings[0])

    # Create collection manually
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )

    # -----------------------------
    # 2️⃣ Upload existing embeddings
    # -----------------------------
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

    # -----------------------------
    # 3️⃣ Wrap with LangChain store
    # -----------------------------
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=clip_embeddings,  # since embeddings are precomputed
    )


    from langchain_groq import ChatGroq
    llm=ChatGroq(model="openai/gpt-oss-120b")

    dense_retriever=vectorstore.as_retriever(search_kwargs={'k':5})
    from langchain.retrievers import EnsembleRetriever

    hybrid_retriever=EnsembleRetriever(
        retrievers=[dense_retriever,sparse_retriever],
        weights=[0.6,0.4]
    )

    from langchain.chains import RetrievalQA
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=hybrid_retriever,
    return_source_documents=True
    )


    def create_multimodel_message(query,retrieved_docs):
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


    from langchain_community.utilities import OpenWeatherMapAPIWrapper
    from langchain_community.tools import OpenWeatherMapQueryRun
    weather_wrapper = OpenWeatherMapAPIWrapper()
    weather=OpenWeatherMapQueryRun(api_wrapper=weather_wrapper)

    tools=[weather]
    llm_with_tools=llm.bind_tools(tools)
    from langgraph.checkpoint.memory import MemorySaver
    memory=MemorySaver()

    def chatbot(state:State):
        response = llm_with_tools.invoke(state['messages'])
        print(f'response from chatbot -- {response}')
        print("\n[Chatbot Node Called] Messages so far:")
        for msg in state['messages']:
            print(msg)
        if response.content:
            print('no tool call -------')
            return state
        return {"messages":[response]}
    
    def rag(state:State):
        '''
        Args:
            query: user input
        To return answers any user questions
        '''
        response=qa.invoke({'query':state['messages'][-1].content})
        print(f'this is the summary------{response}')
        return {'messages':[AIMessage(content=response['result'])]}

    from langgraph.prebuilt import ToolNode
    from langgraph.prebuilt import tools_condition

    graph=StateGraph(State)

    ## node
    graph.add_node("Chatbot",chatbot)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("rag",rag)

    ## Edges

    graph.add_edge(START,"Chatbot")
    graph.add_conditional_edges(
        "Chatbot",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
        {
            'tools': "tools",         # if condition returns True -> tools
            '__end__': "rag"   # if condition returns False -> AnotherNode
        }
    )
    graph.add_edge('tools',END)
    graph.add_edge('rag',END)

    graph_builder=graph.compile(checkpointer=memory)

    ## Invocation

    config = {"configurable": {"thread_id": "1"}}

    output=graph_builder.invoke({'messages':query},config)
    # print(output)
    return output['messages'][-1].content
