import io
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from backend import embed_image, embed_text, invoke_agent


# ---------- A. TEST EMBEDDINGS ----------

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


# ---------- B. TEST INVOKE_AGENT BASIC FLOW ----------

@pytest.fixture
def mock_pdf_file(tmp_path):
    """Create a small fake PDF for testing."""
    from reportlab.pdfgen import canvas
    pdf_path = tmp_path / "test.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "This is a test page.")
    c.showPage()
    c.save()

    # Return file-like object
    return open(pdf_path, "rb")


@patch("backend.embed_text", return_value=np.random.rand(512))
@patch("backend.embed_image", return_value=np.random.rand(512))
@patch("backend.ChatGroq")
@patch("backend.QdrantClient")
@patch("backend.RetrievalQA")
def test_invoke_agent_runs_minimal_flow(
    mock_retrievalqa, mock_qdrant, mock_llm, mock_img_emb, mock_text_emb, mock_pdf_file
):
    # --- Mock Qdrant client ---
    mock_qdrant.return_value = MagicMock()
    mock_qdrant.return_value.recreate_collection.return_value = None
    mock_qdrant.return_value.upsert.return_value = None

    # --- Mock LLM ---
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = MagicMock(content="Mock LLM response")
    mock_llm.return_value = mock_llm_instance

    # --- Mock Retrieval QA ---
    mock_retrieval_instance = MagicMock()
    mock_retrieval_instance.invoke.return_value = {"result": "Mock QA answer"}
    mock_retrievalqa.from_chain_type.return_value = mock_retrieval_instance

    # --- Call invoke_agent ---
    result = invoke_agent(mock_pdf_file, query="What is this PDF about?")
    assert "Mock QA answer" in result


# ---------- C. TEST ERROR HANDLING ----------

def test_embed_image_invalid_input_raises():
    with pytest.raises(Exception):
        embed_image(12345)  # invalid type


# ---------- D. TEST GRAPH BEHAVIOR ----------
@patch("backend.ChatGroq")
@patch("backend.RetrievalQA")
def test_invoke_agent_invokes_llm(mock_retrievalqa, mock_llm, mock_pdf_file):
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = MagicMock(content="Test reply")
    mock_llm.return_value = mock_llm_instance

    mock_retrieval_instance = MagicMock()
    mock_retrieval_instance.invoke.return_value = {"result": "Answer"}
    mock_retrievalqa.from_chain_type.return_value = mock_retrieval_instance

    out = invoke_agent(mock_pdf_file, query="Summarize the document.")
    assert isinstance(out, str)
    assert "Answer" in out
