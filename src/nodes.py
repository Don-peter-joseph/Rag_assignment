from state import State
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from tools import weather
from langchain.chains import RetrievalQA
from process_embeddings import retriever_init

llm=ChatGroq(model="openai/gpt-oss-120b")
tools=[weather]
llm_with_tools=llm.bind_tools(tools)

def chatbot(state:State):
    
    response = llm_with_tools.invoke(state['messages'])
    for msg in state['messages']:
        print(msg)
    if response.content:
        print('no tool call -------')
        return state
    return {"messages":[response]}

def rag(state:State):
    hybrid_retriever=retriever_init(state['uploaded_file'])
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=hybrid_retriever,
    return_source_documents=True
    )
    response=qa.invoke({'query':state['messages'][-1].content})
    print(f'-----------this is the summary\n{response}\n-------------')
    return {'messages':[AIMessage(content=response['result'])]}