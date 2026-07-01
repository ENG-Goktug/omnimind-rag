# main.py
import os
import logging
from src.ingestion.chunker import SemanticChunker
from src.retrieval.hybrid_search import HybridSearchEngine
from src.generation.llm_client import GenerationPipeline

# Configure clean logging output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    print("\n" + "=" * 50)
    print("      Welcome to OmniMind-RAG CLI Terminal      ")
    print("=" * 50 + "\n")

    # 1. Initialize all professional engine components
    logger.info("Initializing system modules...")
    chunker = SemanticChunker()
    search_engine = HybridSearchEngine()
    generation_pipeline = GenerationPipeline()

    # 2. Provide a rigorous, sample technical knowledge base text
    # In a real-world scenario, this can be read directly from files (.pdf, .md, .txt)
    raw_knowledge_data = (
        "Advanced production systems mandate the implementation of zero-trust network architectures. "
        "Every single API request must be rigorously authenticated using signed JSON Web Tokens (JWT). "
        "Database clusters should replicate data asynchronously across multiple geographical regions to guarantee "
        "high availability and disaster recovery. "
        "For local development environments, Docker Compose is highly recommended to orchestrate containerized microservices. "
        "Developers must never hardcode cryptographic keys or production passwords inside the source code repository; "
        "instead, environmental variables or specialized vault services must be utilized."
    )

    print("\n[*] Processing raw knowledge base via Semantic Chunker...")
    semantic_chunks = chunker.create_chunks(raw_knowledge_data)
    print(f"[+] Successfully extracted {len(semantic_chunks)} logically isolated chunks.")

    print("\n[*] Feeding chunks into the Hybrid Search Index (Dense Vector + Sparse BM25)...")
    search_engine.index_documents(semantic_chunks)
    print("[+] Knowledge base indexing complete. System is fully operational.")

    # 3. Interactive Query Loop
    print("\n" + "-" * 50)
    print("Ask OmniMind-RAG anything about the system configuration:")
    print("(Type 'exit' or 'quit' to terminate the session)")
    print("-" * 50 + "\n")

    while True:
        user_query = input("User Question ➔ ").strip()

        if user_query.lower() in ['exit', 'quit']:
            print("\nShutting down OmniMind-RAG pipeline. Goodbye!")
            break

        if not user_query:
            continue

        try:
            print("\n[Retrieving] Searching across hybrid index pools...")
            raw_hits = search_engine.search(user_query, top_k=3)

            print("[Reranking] Re-scoring retrieved artifacts via Cross-Encoder...")
            refined_context = generation_pipeline.rerank_documents(
                query=user_query,
                documents=raw_hits,
                top_n=2
            )

            print("[Generating] Orchestrating response from Local LLM...")
            ai_response = generation_pipeline.generate_answer(
                query=user_query,
                context_documents=refined_context
            )

            print(f"\n🤖 OmniMind-RAG Response:\n{ai_response}\n")
            print("-" * 60 + "\n")

        except Exception as e:
            logger.error(f"An error occurred during pipeline execution: {e}\n")


if __name__ == "__main__":
    main()