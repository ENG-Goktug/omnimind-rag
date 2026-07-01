# OmniMind-RAG: High-Performance Hybrid Information Retrieval & Local Generation System

OmniMind-RAG is a production-grade, local Retrieval-Augmented Generation (RAG) framework built entirely from scratch in Python. Instead of relying on heavy orchestration wrappers, this system implements core LLM engineering patterns natively, demonstrating advanced software architecture, mathematical similarity implementations, and modular design.

## 🚀 Key Engineering Features

- **Semantic Chunking Pipeline**: Replaced standard static character splitting with a dynamic semantic breakpoint algorithm using sentence embedding distances (Cosine Similarity) and sliding percentiles.
- **Hybrid Multi-Stream Retrieval**: Integrates **Dense Vector Search** (via persistent ChromaDB layer) with **Sparse Lexical Search** (via BM25Okapi tokenization modeling) to achieve ultra-precise context matching.
- **Cross-Encoder Reranking**: Leverages a pre-trained `ms-marco-MiniLM` Cross-Encoder backend to re-score raw matched inputs, heavily mitigating LLM hallucinations by supplying compressed, high-value contexts.
- **100% Local & Secure Generation**: Orchestrated natively with a local **Ollama (Llama 3)** daemon instance via optimized REST endpoints, ensuring total data privacy with zero external API dependencies.

---

## 🏗️ System Architecture

The pipeline processes unstructured data through a dual-stage execution model:

1. **Ingestion Pool**: Raw Text ➔ Sentence Tokenization ➔ Embedding Extraction ➔ Cosine Distance Analysis ➔ Dynamic Percentile Split ➔ Dual Indexing (ChromaDB + BM25).
2. **Generation Loop**: User Query ➔ Hybrid Retrieval ➔ Cross-Encoder Reranking ➔ System Prompt Synthesis ➔ Local Ollama Inference ➔ Final Response.

---

## 📁 Directory Structure

```text
omnimind_rag/
│
├── src/
│   ├── config.py             # Centralized configuration management via Pydantic Settings
│   ├── ingestion/            # Semantic processing and text decomposition
│   │   └── chunker.py
│   ├── retrieval/            # Multi-paradigm indexing and search execution
│   │   └── hybrid_search.py
│   └── generation/           # Cross-Encoder scoring and local LLM clients
│       └── llm_client.py
│
├── tests/                    # Automated testing suite powered by PyTest
│   └── test_pipeline.py
│
├── requirements.txt          # Explicit third-party operational dependencies
└── main.py                   # System entrypoint (Interactive CLI Terminal)
⚡ Installation & Local Setup
Prerequisites
Python 3.10+

Ollama App running locally with the Llama3 model installed (ollama run llama3).

1. Clone & Set Up Environment
Bash
git clone [https://github.com/yourusername/omnimind-rag.git](https://github.com/yourusername/omnimind-rag.git)
cd omnimind-rag
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Run Automated Unit Tests
Bash
pytest
4. Launch Interactive Application
Bash
python main.py
🛠️ Tech Stack & Concepts Demonstrated
Languages & Frameworks: Python, NumPy, PyTest, Pydantic.

Vector DB & NLP: ChromaDB, Rank-BM25, Sentence-Transformers (Cross-Encoders/Bi-Encoders).

Design Patterns: Singleton Pattern (Configuration), Tokenization Pipelines, Functional Decomposition, Self-Correction via Zero-Hallucination Prompts.


---

## 📦 GitHub'a Güvenli Yükleme Rehberi (Çok Önemli)

Kodlarımızın içine kişisel bilgilerimizin sızmadığından eminiz, ancak demin konuştuğumuz gibi `data/` klasörünün (ChromaDB'nin yerel dosyaları) ve Python sanal ortamının (`.venv/`) GitHub'a yüklenmesini engellememiz gerekiyor.

1. Projenin en dış dizinine `.gitignore` adında yeni bir dosya aç (PyCharm'da `New -> File` deyip adını tam olarak `.gitignore` yap).
2. İçine şu satırları ekle ve kaydet:
   ```text
   .venv/
   venv/
   __pycache__/
   *.pyc
   data/
   .env
   .idea/