import hashlib
import shutil
import uuid
import json
from pathlib import Path
from typing import Tuple, Optional

from app.config.settings import settings


def generate_unique_filename(original_filename: str) -> str:
    """Generate a unique filename preserving extension."""
    file_ext = Path(original_filename).suffix
    unique_name = f"{uuid.uuid4().hex}{file_ext}"
    return unique_name


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def create_document_directory(document_id: uuid.UUID) -> Path:
    """Create directory structure for a document."""
    doc_dir = settings.DOCUMENTS_DIR / str(document_id)
    pages_dir = doc_dir / "pages"
    
    doc_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    
    return doc_dir


def save_uploaded_file(uploaded_file, document_id: uuid.UUID) -> Tuple[Path, str]:
    """Save uploaded file to document directory."""
    doc_dir = create_document_directory(document_id)
    original_path = doc_dir / "original.pdf"
    
    # Save the uploaded file
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    
    return original_path, str(original_path.relative_to(settings.BASE_DATA_DIR))


def serialize_pydantic_model(model):
    """Serialize Pydantic model to JSON string with Pydantic v2 compatibility."""
    # For Pydantic v2 - use model_dump_json
    if hasattr(model, 'model_dump_json'):
        try:
            return model.model_dump_json(indent=2)
        except Exception as e:
            print(f"Warning: model_dump_json failed: {e}")
            # Fall through to next method
    
    # Ultimate fallback - convert to dict and use json.dumps
    import json
    try:
        # Try model_dump() first (Pydantic v2)
        if hasattr(model, 'model_dump'):
            data = model.model_dump()
        # Then try dict() (Pydantic v1)
        elif hasattr(model, 'dict'):
            data = model.dict()
        else:
            data = str(model)
        
        return json.dumps(data, indent=2, default=str)
    except Exception as e:
        print(f"Error serializing model: {e}")
        return "{}"


def save_metadata(metadata, document_id: uuid.UUID) -> Path:
    """Save metadata to JSON file."""
    doc_dir = settings.DOCUMENTS_DIR / str(document_id)
    metadata_path = doc_dir / "metadata.json"
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json_str = serialize_pydantic_model(metadata)
        f.write(json_str)
    
    return metadata_path