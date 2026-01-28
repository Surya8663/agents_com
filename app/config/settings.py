import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    # Server settings
    APP_NAME: str = "Document Intelligence System"
    API_VERSION: str = "v1"
    DEBUG: bool = False
    
    # File storage settings
    BASE_DATA_DIR: Path = Path("data")
    DOCUMENTS_DIR: Path = BASE_DATA_DIR / "documents"
    
    # ✅ ADD THIS: YOLOv8 model path for Phase 2
    LAYOUT_MODEL_PATH: Path = Path("yolov8n.pt")
    
    # File size limits (10MB default)
    MAX_FILE_SIZE_MB: int = 10
    MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
    
    # PDF processing settings
    PDF_IMAGE_DPI: int = 150
    PDF_IMAGE_FORMAT: str = "PNG"
    
    # Poppler path for Windows
    POPPLER_PATH: Optional[str] = None
    
    # Phase 4: LLM Settings
    LLM_BASE_URL: Optional[str] = "http://localhost:11434/v1"
    LLM_MODEL: Optional[str] = "qwen2.5:1.5b"
    
    # Phase 4: Vector Database
    QDRANT_HOST: Optional[str] = "localhost"
    QDRANT_PORT: Optional[int] = 6333
    
    # Phase 4: GPU Settings (optional)
    CUDA_VISIBLE_DEVICES: Optional[str] = None
    
    # Phase 4: Embedding settings
    EMBEDDING_MODEL: Optional[str] = "all-MiniLM-L6-v2"
    EMBEDDING_SIZE: Optional[int] = 384
    
    # Create directories if they don't exist
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        # ✅ Also create models directory
        self.LAYOUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"  # Allow extra fields from .env
    )


settings = Settings()