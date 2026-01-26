import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.config.settings import settings

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Document Intelligence System" in data["service"]


def test_service_info():
    """Test service information endpoint."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["phase"] == 1
    assert "Document Ingestion" in data["description"]


def test_upload_invalid_file():
    """Test uploading a non-PDF file."""
    files = {"file": ("test.txt", b"Not a PDF", "text/plain")}
    response = client.post("/ingest/upload", files=files)
    assert response.status_code == 400
    assert "File must be a PDF" in response.json()["detail"]


def test_upload_empty_file():
    """Test uploading an empty file."""
    files = {"file": ("empty.pdf", b"", "application/pdf")}
    response = client.post("/ingest/upload", files=files)
    assert response.status_code == 400
    assert "File is empty" in response.json()["detail"]


def test_upload_valid_pdf():
    """Test uploading a valid PDF file."""
    # Create a simple PDF for testing using reportlab
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        import io
        
        # Create a PDF in memory
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        
        # Add some text to the first page
        c.drawString(100, 750, "Test PDF Page 1")
        c.showPage()
        
        # Add a second page
        c.drawString(100, 750, "Test PDF Page 2")
        c.showPage()
        
        c.save()
        
        # Reset buffer position
        buffer.seek(0)
        pdf_content = buffer.getvalue()
        
        # Upload the PDF
        files = {"file": ("test.pdf", pdf_content, "application/pdf")}
        response = client.post("/ingest/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert data["success"] is True
        assert data["total_pages"] == 2
        assert data["pages_processed"] == 2
        assert "document_id" in data
        assert "metadata_path" in data
        
        # Verify files were created
        document_id = data["document_id"]
        doc_dir = settings.DOCUMENTS_DIR / document_id
        
        assert doc_dir.exists()
        assert (doc_dir / "original.pdf").exists()
        assert (doc_dir / "metadata.json").exists()
        assert (doc_dir / "pages").exists()
        
        # Verify metadata
        metadata_path = doc_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        assert metadata["document_id"] == document_id
        assert metadata["filename"] == "test.pdf"
        assert metadata["total_pages"] == 2
        assert len(metadata["pages"]) == 2
        
        # Verify page images
        for page_num in range(1, 3):
            page_image = doc_dir / "pages" / f"page_{page_num}.png"
            assert page_image.exists()
            
            # Verify it's a valid image
            with Image.open(page_image) as img:
                assert img.format == "PNG"
                assert img.size[0] > 0  # Width
                assert img.size[1] > 0  # Height
        
        # Clean up test data
        import shutil
        shutil.rmtree(doc_dir)
        
    except ImportError:
        pytest.skip("reportlab not installed for PDF generation")
    except Exception as e:
        # Clean up on any error
        if 'doc_dir' in locals() and doc_dir.exists():
            import shutil
            shutil.rmtree(doc_dir)
        raise


def test_document_status():
    """Test document status endpoint."""
    # First upload a document
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        import io
        
        # Create a simple PDF
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.drawString(100, 750, "Status Test PDF")
        c.showPage()
        c.save()
        buffer.seek(0)
        pdf_content = buffer.getvalue()
        
        # Upload the PDF
        files = {"file": ("status_test.pdf", pdf_content, "application/pdf")}
        response = client.post("/ingest/upload", files=files)
        assert response.status_code == 200
        
        data = response.json()
        document_id = data["document_id"]
        
        # Check status
        response = client.get(f"/ingest/status/{document_id}")
        assert response.status_code == 200
        status_data = response.json()
        assert status_data["status"] == "completed"
        assert status_data["document_id"] == document_id
        
        # Clean up
        doc_dir = settings.DOCUMENTS_DIR / document_id
        if doc_dir.exists():
            import shutil
            shutil.rmtree(doc_dir)
            
    except ImportError:
        pytest.skip("reportlab not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])