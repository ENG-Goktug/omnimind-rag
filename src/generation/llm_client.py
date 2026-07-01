# src/generation/llm_client.py
import logging
from typing import List, Dict, Any
import requests
from sentence_transformers import CrossEncoder
from src.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GenerationPipeline:
    """
    Manages the Reranking of retrieved documents and handles orchestrated
    communication with the local Ollama LLM instance.
    """

    def __init__(self):
        # Load the Cross-Encoder model to compute deep relevancy scores
        logger.info(f"Loading Reranker model: {settings.RERANKER_MODEL}...")
        self.reranker = CrossEncoder(settings.RERANKER_MODEL)

    def rerank_documents(self, query: str, documents: List[Dict[str, Any]], top_n: int = 2) -> List[str]:
        """
        Evaluates query-document pairs using a Cross-Encoder and returns the top_n most relevant texts.
        """
        if not documents:
            return []

        # Prepare pairs for the Cross-Encoder: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc["content"]] for doc in documents]

        # Predict relevancy scores (higher score means more relevant)
        scores = self.reranker.predict(pairs)

        # Pair documents with their calculated scores
        for idx, score in enumerate(scores):
            documents[idx]["rerank_score"] = float(score)

        # Sort documents dynamically by score in descending order
        sorted_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        # Extract and return the raw text of top-performing items
        return [doc["content"] for doc in sorted_docs[:top_n]]

    def generate_answer(self, query: str, context_documents: List[str]) -> str:
        """
        Constructs a secure system prompt and streams verification requests to the local Ollama instance.
        """
        # Combine the high-value context pieces into a single structured string
        context_str = "\n".join([f"- {doc}" for doc in context_documents])

        # Crafting a rigorous system prompt to prevent LLM hallucinations
        system_prompt = (
            "You are an expert technical assistant. Answer the user query strictly using the provided context below. "
            "If the context does not contain the answer, state clearly that you do not possess that information. "
            "Do not manufacture or extrapolate answers beyond the text provided.\n\n"
            f"Context:\n{context_str}"
        )

        # Target endpoint defined by Ollama local API specifications
        api_url = f"{settings.OLLAMA_BASE_URL}/api/generate"

        payload = {
            "model": settings.LLM_MODEL,
            "prompt": query,
            "system": system_prompt,
            "stream": False  # Setting to False for a clean, non-streaming string response block
        }

        try:
            logger.info(f"Sending context-enriched payload to Ollama ({settings.LLM_MODEL})...")
            response = requests.post(api_url, json=payload, timeout=60)
            response.raise_for_status()

            response_data = response.json()
            return response_data.get("response", "No response content generated.")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to communicate with local Ollama service: {e}")
            return "Error: Unable to generate an answer via local LLM backend."


# Standalone end-to-end execution testing block
if __name__ == "__main__":
    # Ensure your Ollama application is running locally before running this block!
    pipeline = GenerationPipeline()

    # Simulating raw hits originating from our hybrid search component
    mock_retrieved_hits = [
        {"content": "Production servers require secure SSH key pairs and deactivated root logins."},
        {"content": "Python decorators extend behavior of a function without modifying its core structure."}
    ]

    test_query = "What is the security requirement for production servers?"

    # 1. Rerank raw matches
    refined_context = pipeline.rerank_documents(query=test_query, documents=mock_retrieved_hits, top_n=1)
    print(f"\nHighly Refined Context selected by Reranker:\n{refined_context}\n")

    # 2. Fire orchestrated payload to Local LLM
    llm_response = pipeline.generate_answer(query=test_query, context_documents=refined_context)
    print(f"--- LLM Response ---\n{llm_response}")