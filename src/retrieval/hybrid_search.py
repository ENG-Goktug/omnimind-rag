# src/retrieval/hybrid_search.py
import logging
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from src.config import settings

# Setup basic logging for clean debugging output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    A Production-grade Search Engine combining Dense Retrieval (Vector Similarity via ChromaDB)
    and Sparse Retrieval (Keyword matching via BM25Okapi) to maximize search accuracy.
    """

    def __init__(self):
        # Initialize persistent ChromaDB local storage
        self.chroma_client = chromadb.PersistentClient(path=settings.DB_PATH)

        # Use HuggingFace embedding function compatible with ChromaDB
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL
        )

        # Create or fetch the vector collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="omnimind_knowledge_base",
            embedding_function=self.embedding_fn
        )

        # Internal placeholders for BM25 sparse index components
        self.bm25 = None
        self.raw_documents: List[str] = []

    def index_documents(self, documents: List[str]) -> None:
        """
        Populates both Vector Database and BM25 index with provided document chunks.
        """
        if not documents:
            logger.warning("Empty document list provided for indexing.")
            return

        self.raw_documents = documents
        logger.info(f"Indexing {len(documents)} document chunks into Hybrid Search Engine...")

        # 1. Indexing into Vector DB (ChromaDB)
        # Generating deterministic string IDs for each chunk
        ids = [f"doc_chunk_{i}" for i in range(len(documents))]
        self.collection.add(
            documents=documents,
            ids=ids
        )
        logger.info("Dense vector storage generation complete.")

        # 2. Indexing into Sparse Index (BM25)
        # Tokenizing documents by lowercase splitting for simple lexical mapping
        tokenized_corpus = [doc.lower().split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Sparse keyword BM25 index generation complete.")

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Executes a hybrid retrieval logic, extracting best hits from both vector and keyword models.
        """
        if not self.raw_documents or self.bm25 is None:
            raise ValueError("Search engine index is empty. Please run index_documents() first.")

        # --- Stream 1: Dense Vector Retrieval ---
        vector_results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        # Flatten ChromaDB nested response structures safely
        dense_hits = vector_results['documents'][0] if vector_results['documents'] else []

        # --- Stream 2: Sparse Keyword Retrieval ---
        tokenized_query = query.lower().split(" ")
        # Fetch high-scoring sentences directly from BM25 backend
        sparse_hits = self.bm25.get_top_n(tokenized_query, self.raw_documents, n=top_k)

        # --- Stream 3: Hybrid Deduping & Merging ---
        # We consolidate hits from both paradigms, giving an elegant unified interface
        combined_hits = list(dict.fromkeys(dense_hits + sparse_hits))

        # Structure payload to maintain professional modular contract downstream
        formatted_results = [{"content": hit, "source": "hybrid_index"} for hit in combined_hits[:top_k]]
        return formatted_results


# Standalone execution verification demo
if __name__ == "__main__":
    # Mock data to simulate crawled knowledge base items
    knowledge_base_mock = [
        "Python decorators extend behavior of a function without modifying its core structure.",
        "Production servers require secure SSH key pairs and deactivated root logins.",
        "To compile modern machine learning apps, high performance NVIDIA GPUs are ideal choices.",
        "Decorators can accept arguments to dynamically configure runtime pipeline layers."
    ]

    engine = HybridSearchEngine()
    engine.index_documents(knowledge_base_mock)

    # Test Query highlighting semantic inference over literal matches
    test_query = "Tell me about function decorators in python"
    search_results = engine.search(test_query, top_k=2)

    print(f"\n--- Search results for query: '{test_query}' ---\n")
    for rank, result in enumerate(search_results, 1):
        print(f"Rank {rank}: {result['content']}")