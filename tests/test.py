import io
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from src.agent import invoke_graph
from src.process_embeddings import embed_image, embed_text


# ---------- TEST EMBEDDINGS ----------

def test_embed_image_returns_normalized_vector():
    # Create dummy image
    img = Image.new("RGB", (32, 32), color="white")
    vec = embed_image(img)
    assert isinstance(vec, np.ndarray)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)

def test_embed_text_returns_normalized_vector():
    text = "Test text for embeddings"
    vec = embed_text(text)
    assert isinstance(vec, np.ndarray)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


# ---------- TEST INVOKE_AGENT BASIC FLOW ----------

@pytest.fixture
def mock_pdf_file(tmp_path):
    """Create a small fake PDF for testing."""
    from reportlab.pdfgen import canvas
    pdf_path = tmp_path / "test.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "This is a test page.")
    c.showPage()
    c.save()

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    return file_bytes



# ---------- TEST ERROR HANDLING ----------

def test_embed_image_invalid_input_raises():
    with pytest.raises(Exception):
        embed_image(12345)  # invalid type


# ---------- TEST GRAPH BEHAVIOR ----------
@patch("src.nodes.ChatGroq")
@patch("src.nodes.RetrievalQA")
def test_invoke_agent_invokes_llm(mock_retrievalqa, mock_llm, mock_pdf_file):
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = MagicMock(content="Test reply")
    mock_llm.return_value = mock_llm_instance

    mock_retrieval_instance = MagicMock()
    mock_retrieval_instance.invoke.return_value = {"result": "Answer"}
    mock_retrievalqa.from_chain_type.return_value = mock_retrieval_instance

    out = invoke_graph(mock_pdf_file, query="Summarize the document.")
    assert isinstance(out, str)
    assert "Answer" in out
