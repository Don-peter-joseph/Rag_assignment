from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from state import State
from nodes import chatbot,rag
from tools import weather


def invoke_graph(uploaded_file,query):

    memory=MemorySaver()
    graph=StateGraph(State)
    tools=[weather]
    ## node
    graph.add_node("Chatbot",chatbot)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("rag",rag)

    ## Edges

    graph.add_edge(START,"Chatbot")
    graph.add_conditional_edges(
        "Chatbot",
        tools_condition,
        {
            'tools': "tools",        
            '__end__': "rag"  
        }
    )
    graph.add_edge('tools',END)
    graph.add_edge('rag',END)

    graph_builder=graph.compile(checkpointer=memory)

    ## Invocation
    config = {"configurable": {"thread_id": "1"}}

    output=graph_builder.invoke({'messages':query,'uploaded_file':uploaded_file},config)
    return output['messages'][-1].content
