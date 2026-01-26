import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Server settings
    APP_NAME: str = "Document Intelligence System - Phase 1"
    API_VERSION: str = "v1"
    DEBUG: bool = False
    
    # File storage settings
    BASE_DATA_DIR: Path = Path("data")
    DOCUMENTS_DIR: Path = BASE_DATA_DIR / "documents"
    
    # File size limits (10MB default)
    MAX_FILE_SIZE_MB: int = 10
    MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
    
    # PDF processing settings
    PDF_IMAGE_DPI: int = 150
    PDF_IMAGE_FORMAT: str = "PNG"
    
    # Poppler path for Windows
    POPPLER_PATH: Optional[str] = None
    
    # Create directories if they don't exist
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"


settings = Settings()