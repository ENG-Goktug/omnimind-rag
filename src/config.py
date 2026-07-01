# src/config.py
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "OmniMind-RAG"

    # Yerel LLM ve Embedding Ayarları
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3"  # Ollama'da indirdiğin model ismi
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Metinleri vektöre çeviren hafif/güçlü model
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Sonuçları yeniden sıralayacak model

    # Chunking (Metin Parçalama) Ayarları
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Veritabanı Kayıt Yolu
    DB_PATH: str = os.path.join(os.getcwd(), "data", "chroma_db")

    class Config:
        env_file = ".env"


# Proje genelinde tek bir nesne üzerinden ayarlara erişeceğiz (Singleton Pattern)
settings = Settings()