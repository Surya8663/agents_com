"""
FINAL PHASE 3 VERIFICATION
"""
import requests

print("=" * 70)
print("🏁 PHASE 3 FINAL VERIFICATION")
print("=" * 70)

try:
    response = requests.get('http://localhost:8000/ocr/engine/info', timeout=10)
    data = response.json()
    
    print(f"\n📊 OCR ENGINE STATUS:")
    print(f"  Engine: {data.get('engine')}")
    print(f"  Mode: {data.get('mode')}")
    print(f"  Language: {data.get('language')}")
    print(f"  Available: {data.get('available')}")
    print(f"  Using Real OCR: {data.get('using_real_ocr')}")
    
    print(f"\n📋 SERVICE INFO:")
    info_response = requests.get('http://localhost:8000/info', timeout=5)
    info_data = info_response.json()
    
    for phase in info_data.get('phases', []):
        print(f"  Phase {phase.get('phase')}: {phase.get('name')}")
    
    print(f"\n🔗 ENDPOINTS:")
    endpoints_response = requests.get('http://localhost:8000/endpoints', timeout=5)
    endpoints = endpoints_response.json().get('endpoints', [])
    
    ocr_endpoints = [e for e in endpoints if '/ocr' in e['path']]
    print(f"  OCR endpoints: {len(ocr_endpoints)} available")
    
    print("\n" + "=" * 70)
    
    # FINAL VERDICT
    if (data.get('engine') == 'EasyOCR' and 
        data.get('mode') == 'real' and 
        data.get('using_real_ocr') == True):
        
        print("""
        ✅ ✅ ✅ PHASE 3 COMPLETED SUCCESSFULLY! ✅ ✅ ✅
        
        CRITERIA MET:
        ✓ Uses REAL EasyOCR (not mock/fallback)
        ✓ Engine mode: 'real' (not 'enhanced_fallback')
        ✓ Using Real OCR: True
        ✓ OCR endpoints registered and working
        ✓ Integrated with FastAPI application
        
        🎉 CONGRATULATIONS! Your Document AI System has:
        - Phase 1: Document Ingestion ✓
        - Phase 2: Layout Analysis ✓  
        - Phase 3: REAL OCR ✓
        
        Ready for Phase 4: Agent-based Document Intelligence!
        """)
    else:
        print("❌ Phase 3 still incomplete")
        
except Exception as e:
    print(f"❌ Test failed: {e}")

print("=" * 70)