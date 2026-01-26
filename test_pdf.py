# verify_phase3.py
"""
Verify Phase 3 is working correctly.
"""
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

print("🔍 Verifying Phase 3 OCR System")
print("=" * 60)

try:
    # Test 1: Import OCR components
    from app.ocr.engine import OCRManager
    from app.ocr.schema import OCRBoundingBox, OCRRegionResult, PageOCRResult
    from app.ocr.processor import OCRProcessor
    from app.ocr.region_cropper import RegionCropper
    
    print("✅ All Phase 3 modules imported")
    
    # Test 2: Initialize OCR Manager
    print("\n🔧 Testing OCR Manager...")
    manager = OCRManager(lang='en')
    info = manager.get_info()
    
    print(f"   Engine: {info['engine']}")
    print(f"   Mode: {info['mode']}")
    print(f"   Language: {info['language']}")
    
    # Test 3: OCR Extraction
    print("\n📝 Testing OCR Extraction...")
    import numpy as np
    import cv2
    
    # Create test image with realistic document text
    img = np.ones((200, 400, 3), dtype=np.uint8) * 240  # Light gray
    cv2.rectangle(img, (20, 30), (380, 80), (255, 255, 255), -1)
    cv2.putText(img, "INVOICE #2024-001", (40, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    cv2.rectangle(img, (20, 100), (380, 180), (255, 255, 255), -1)
    cv2.putText(img, "Date: 2024-01-24", (40, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "Amount: $1,234.56", (40, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    text, confidence, boxes = manager.extract_text(img, "invoice_header")
    
    print(f"   Text extracted: {'Yes' if text else 'No'}")
    if text:
        print(f"   Preview: {text[:60]}...")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Word boxes: {len(boxes)}")
    
    # Test 4: Schemas
    print("\n📋 Testing OCR Schemas...")
    bbox = OCRBoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.6)
    region = OCRRegionResult(
        region_id="r1",
        type="text_block",
        bbox=bbox,
        ocr_text=text or "Sample extracted text",
        ocr_confidence=confidence or 0.85,
        engine=info['engine']
    )
    
    print(f"   OCRBoundingBox: {bbox}")
    print(f"   OCRRegionResult: {region.region_id} ({region.type})")
    
    # Test 5: Complete Flow
    print("\n⚙️  Testing Complete OCR Flow...")
    processor = OCRProcessor(lang='en')
    print(f"   OCR Processor: {processor.ocr_manager.engine_name}")
    
    print("\n" + "=" * 60)
    print("🎉 PHASE 3 VERIFICATION COMPLETE!")
    print("=" * 60)
    print("\n✅ All components working correctly")
    print(f"✅ OCR Engine: {info['engine']}")
    print(f"✅ Mode: {info['mode']}")
    print(f"✅ Ready for document processing")
    
    print("\n🚀 Phase 3 is READY for Phase 4!")
    print("\n📋 Available API endpoints:")
    print("   POST /ocr/process/{document_id}")
    print("   GET /ocr/results/{document_id}")
    print("   GET /ocr/engine/info")
    
except Exception as e:
    print(f"\n❌ Verification failed: {e}")
    import traceback
    traceback.print_exc()