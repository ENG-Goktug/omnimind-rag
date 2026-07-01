# src/ingestion/chunker.py
import re
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import settings


class SemanticChunker:
    """
    Advanced text chunking mechanism that splits text based on semantic transitions
    and embedding distance rather than static character counts.
    """

    def __init__(self):
        # Load the pre-trained embedding model specified in config (downloads automatically on first run)
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits the raw text into individual sentences using regex."""
        # Splits text at sentence boundaries (.!? followed by whitespace)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculates the cosine similarity between two vectors."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))

    def create_chunks(self, text: str, threshold_percentile: float = 60.0) -> List[str]:
        """
        Groups sentences into chunks by identifying semantic breakpoints.
        threshold_percentile: Sensitivity coefficient to determine the distance threshold for splitting.
        """
        sentences = self._split_into_sentences(text)
        if len(sentences) < 2:
            return sentences

        # 1. Generate embeddings for all extracted sentences
        embeddings = self.model.encode(sentences)

        # 2. Calculate cosine similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # 3. Determine the dynamic breakpoint threshold using percentiles
        # Lower similarity means higher semantic distance (potential breakpoint)
        distances = [1 - sim for sim in similarities]
        breakpoint_threshold = np.percentile(distances, threshold_percentile)

        # 4. Construct chunks based on the computed semantic distances
        chunks = []
        current_chunk = sentences[0]

        for i in range(len(sentences) - 1):
            # If the semantic distance exceeds the threshold, trigger a split
            if distances[i] > breakpoint_threshold:
                chunks.append(current_chunk)
                current_chunk = sentences[i + 1]
            else:
                # If the semantic flow is continuous, append the sentence to the current chunk
                current_chunk += " " + sentences[i + 1]

        chunks.append(current_chunk)
        return chunks


# Standalone demo execution block for testing verification
if __name__ == "__main__":
    test_text = (
        "Artificial intelligence technologies have advanced rapidly in recent years. Especially deep learning models "
        "have triggered a major breakthrough. Today, computers can perform cognitive reasoning like humans. "
        "On the other hand, I haven't decided what to eat for dinner yet. Ordering a large pizza might be a good "
        "idea, or I could just cook some pasta at home. "
        "In software engineering, writing clean code is critical for long-term project sustainability. "
        "Development teams that tolerate spaghetti code eventually face unmaintainable codebases."
    )

    chunker = SemanticChunker()
    result_chunks = chunker.create_chunks(test_text)

    print(f"\n--- Total of {len(result_chunks)} Semantic Chunks Created ---\n")
    for idx, chunk in enumerate(result_chunks, 1):
        print(f"Chunk {idx}:\n{chunk}\n{'-' * 20}")