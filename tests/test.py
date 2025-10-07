import pytest
import sys
sys.path.append("/Volumes/Mac_SSD1/GEN AI/assignment/src")
from unittest.mock import patch, MagicMock
from reportlab.pdfgen import canvas
import io
from langchain_core.messages import AIMessage

@pytest.fixture
def sample_pdf_bytes():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(100, 750, "Test PDF")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

@pytest.fixture
def sample_query():
    return [{"type": "human", "content": "Hello"}]

@pytest.fixture
def mock_llm_response():
    return AIMessage(content="Mocked response")

@pytest.fixture
def mock_tool_response():
    return AIMessage(content="Mocked weather response")

@pytest.fixture
def mock_rag_response():
    return {"messages":[AIMessage(content="Mocked RAG answer")]}

# Patch the full rag function in nodes.py
@patch("src.agent.rag")
@patch("src.nodes.llm_with_tools")
def test_invoke_rag_graph(mock_llm_with_tools, mock_rag, sample_pdf_bytes, sample_query, mock_rag_response, mock_llm_response):
    from src.agent import invoke_graph

    # Make llm_with_tools.invoke return a mocked response
    mock_llm_with_tools.invoke.return_value = mock_llm_response

    # Make rag return mocked RAG response
    mock_rag.return_value = mock_rag_response

    # Run the graph
    output = invoke_graph(sample_pdf_bytes, sample_query)
    print("OUTPUT:", output)

    # Assertions
    assert output in ["Mocked response", "Mocked RAG answer"]
    mock_rag.assert_called_once()


def mock_weather_tool(*args, **kwargs):
    """Mock weather tool for testing"""
    return AIMessage(content="Mocked weather response")

@patch("src.agent.chatbot")
@patch("src.agent.tools_condition", lambda state: "tools")  # return valid branch key
def test_invoke_tools_branch(mock_chatbot, sample_pdf_bytes, sample_query):
    import src.agent as agent

    mock_chatbot.return_value = {"messages":[AIMessage(content="Mocked weather response")]}

    def mock_weather_tool(*args, **kwargs):
        """Mock weather tool for testing"""
        from langchain_core.messages import AIMessage
        return AIMessage(content="Mocked weather response")

    agent.weather = mock_weather_tool

    output = agent.invoke_graph(sample_pdf_bytes, sample_query)
    assert output == "Mocked weather response"