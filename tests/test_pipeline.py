# tests/test_pipeline.py
import pytest
import numpy as np
from src.ingestion.chunker import SemanticChunker


def test_split_into_sentences():
    """Tests if the sentence splitting regex functions correctly."""
    chunker = SemanticChunker()
    text = "Hello world. This is a test sentence! Is everything working?"
    sentences = chunker._split_into_sentences(text)

    assert len(sentences) == 3
    assert sentences[0] == "Hello world."
    assert sentences[1] == "This is a test sentence!"
    assert sentences[2] == "Is everything working?"


def test_cosine_similarity():
    """Tests the mathematical accuracy of the internal cosine similarity function."""
    chunker = SemanticChunker()
    # Perfect alignment
    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 0, 0])
    assert pytest.approx(chunker._cosine_similarity(v1, v2), 0.01) == 1.0

    # Orthogonal vectors (completely different)
    v3 = np.array([0, 1, 0])
    assert pytest.approx(chunker._cosine_similarity(v1, v3), 0.01) == 0.0


def test_create_chunks_single_sentence():
    """Tests if a single sentence is returned as-is without crashing."""
    chunker = SemanticChunker()
    text = "Just a single short sentence."
    chunks = chunker.create_chunks(text)

    assert len(chunks) == 1
    assert chunks[0] == text