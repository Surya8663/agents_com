import tempfile
from pathlib import Path
from typing import List, Optional
import sys

from pdf2image import convert_from_path
from PIL import Image

from app.config.settings import settings
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.schema import PageMetadata


class PageProcessor:
    """Processes individual PDF pages."""
    
    def __init__(self, pdf_path: Path, document_id: str):
        self.pdf_path = pdf_path
        self.document_id = document_id
        self.pdf_loader = PDFLoader()
        
        # Get document directories
        self.doc_dir = settings.DOCUMENTS_DIR / document_id
        self.pages_dir = self.doc_dir / "pages"
    
    def process_page(self, page_num: int) -> PageMetadata:
        """Process a single page: extract text and convert to image."""
        # Extract native text
        native_text = self.pdf_loader.extract_native_text(self.pdf_path, page_num)
        native_text_available = native_text is not None
        
        # Generate image filename and path
        image_filename = f"page_{page_num}.png"
        image_path = self.pages_dir / image_filename
        
        # Convert page to image
        self._convert_page_to_image(page_num, image_path)
        
        # Create page metadata
        return PageMetadata(
            page_number=page_num,
            image_path=str(image_path.relative_to(settings.BASE_DATA_DIR)),
            native_text_available=native_text_available,
            native_text_length=len(native_text) if native_text_available else 0,
            native_text_content=native_text if native_text_available else None
        )
    
    def _convert_page_to_image(self, page_num: int, output_path: Path):
        """Convert a PDF page to image using pdf2image."""
        try:
            from pdf2image import convert_from_path
            
            # Prepare conversion arguments
            convert_args = {
                "dpi": settings.PDF_IMAGE_DPI,
                "first_page": page_num,
                "last_page": page_num,
                "fmt": settings.PDF_IMAGE_FORMAT.lower(),
            }
            
            # Add poppler_path if specified in settings
            if hasattr(settings, 'POPPLER_PATH') and settings.POPPLER_PATH:
                # Convert forward slashes to backslashes for Windows
                poppler_path = str(settings.POPPLER_PATH).replace('/', '\\')
                convert_args["poppler_path"] = poppler_path
                print(f"Using poppler_path: {poppler_path}")
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Method 1: Direct save (most reliable)
            try:
                images = convert_from_path(
                    str(self.pdf_path),
                    **convert_args
                )
                
                if images and len(images) > 0:
                    # Save the image directly
                    images[0].save(output_path, 'PNG')
                    print(f"✓ Direct save: Created image for page {page_num} at {output_path}")
                    return
                else:
                    raise ValueError(f"No images returned for page {page_num}")
                    
            except Exception as e1:
                print(f"Direct save failed: {e1}")
                
                # Method 2: Try with output_folder parameter
                try:
                    # Create a temporary filename pattern
                    temp_pattern = f"page_{page_num}"
                    
                    images = convert_from_path(
                        str(self.pdf_path),
                        **convert_args,
                        output_folder=str(output_path.parent),
                        output_file=temp_pattern,
                        paths_only=False  # Let pdf2image handle the saving
                    )
                    
                    # Check if file was created with expected pattern
                    expected_files = list(output_path.parent.glob(f"{temp_pattern}*.png"))
                    if expected_files:
                        print(f"✓ Output folder method: Created {len(expected_files)} image(s)")
                        # If multiple files, rename the first one
                        if expected_files[0].name != output_path.name:
                            expected_files[0].rename(output_path)
                        return
                        
                except Exception as e2:
                    print(f"Output folder method failed: {e2}")
                    
                    # Method 3: Last resort - save with a different approach
                    try:
                        images = convert_from_path(
                            str(self.pdf_path),
                            **convert_args
                        )
                        
                        if images:
                            # Save with PIL
                            from PIL import Image
                            images[0].save(output_path, 'PNG')
                            print(f"✓ PIL fallback: Saved image for page {page_num}")
                            return
                            
                    except Exception as e3:
                        raise RuntimeError(f"All conversion methods failed: {e1}, {e2}, {e3}")
            
            # If we get here, no method worked
            raise RuntimeError(f"Failed to convert page {page_num} to image")
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert page {page_num} to image: {str(e)}")
    
    def process_all_pages(self, total_pages: int) -> List[PageMetadata]:
        """Process all pages in the document."""
        pages_metadata = []
        
        for page_num in range(1, total_pages + 1):
            try:
                page_metadata = self.process_page(page_num)
                pages_metadata.append(page_metadata)
                print(f"Successfully processed page {page_num}/{total_pages}")
            except Exception as e:
                # Log error but continue with other pages
                print(f"Error processing page {page_num}: {str(e)}")
                raise  # Re-raise for now
        
        return pages_metadata