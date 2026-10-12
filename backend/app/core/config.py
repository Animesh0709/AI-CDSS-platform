from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "AI-CDSS: Clinical Decision Support System"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    
    DATABASE_URL: str = "sqlite:///./ai_cdss.db"
    
    JWT_SECRET_KEY: str = "supersecretkey_change_in_production_ai_cdss_2026"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    MODEL_PATH: str = "./ml/models"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DB_PATH: str = "./rag/vector_db"
    LLM_PROVIDER: str = ""
    LLM_API_KEY: str = ""
    LLM_MODEL: str = ""
    RANDOM_SEED: int = 42
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 10
    TESSERACT_CMD: str = ""  # Optional: path to tesseract executable
    POPPLER_PATH: str = ""  # Optional: folder containing pdftoppm for PDF OCR

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_label(cls, value):
        """Allow common deployment labels such as DEBUG=release in local .env files."""
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "debug", "development"}
        return value

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
