# quick_complete_test.py
import requests
import time

print("🚀 Quick Complete System Test")
print("=" * 60)

# 1. Upload a simple test document
print("1. Uploading test document...")
from reportlab.pdfgen import canvas
from io import BytesIO

buffer = BytesIO()
c = canvas.Canvas(buffer)
c.drawString(100, 700, "Test Invoice #INV-TEST-123")
c.drawString(100, 680, "Date: Today")
c.drawString(100, 660, "Total: $100.00")
c.save()

buffer.seek(0)
files = {'file': ('quick_test.pdf', buffer, 'application/pdf')}

response = requests.post("http://localhost:8000/ingest/upload", files=files, timeout=30)
if response.status_code == 200:
    doc_id = response.json().get('document_id')
    print(f"✅ Document uploaded: {doc_id}")
else:
    print(f"❌ Upload failed: {response.status_code}")
    exit()

# 2. Run Phase 2 (Layout)
print("\n2. Running layout analysis...")
response = requests.post(f"http://localhost:8000/layout/analyze/{doc_id}", timeout=30)
print(f"   Status: {response.json().get('message', 'N/A')}")

# Wait a moment
time.sleep(3)

# 3. Run Phase 3 (OCR)
print("\n3. Running OCR processing...")
response = requests.post(f"http://localhost:8000/ocr/process/{doc_id}", timeout=30)
print(f"   Status: {response.json().get('message', 'N/A')}")

# Wait a moment
time.sleep(3)

# 4. Run Phase 4 (Agents)
print("\n4. Running agents...")
response = requests.post(f"http://localhost:8000/agents/run/{doc_id}", timeout=30)
print(f"   Status: {response.json().get('message', 'N/A')}")

# Wait for agents
print("\n5. Waiting for agents (30 seconds)...")
time.sleep(30)

# 5. Check results
print("\n6. Checking results...")
response = requests.get(f"http://localhost:8000/agents/result/{doc_id}", timeout=10)
if response.status_code == 200:
    results = response.json()
    
    print(f"   Status: {results.get('status', 'unknown')}")
    print(f"   Agents executed: {results.get('agents_executed', [])}")
    
    if results.get('status') == 'completed':
        # Check Text Agent
        text_analysis = results.get('text_analysis', {})
        print(f"   Document type: {text_analysis.get('document_type', 'unknown')}")
        
        # Check if real LLM was used
        key_value_pairs = text_analysis.get('key_value_pairs', {})
        if key_value_pairs:
            print(f"🎉 REAL LLM EXTRACTION WORKING!")
            print(f"   Extracted {len(key_value_pairs)} fields:")
            for key, value in key_value_pairs.items():
                print(f"     • {key}: {value}")
        else:
            print("⚠️ No fields extracted")
            
            # Check server logs for LLM messages
            print("\n💡 Check server terminal for:")
            print("   - 'Text Agent calling LLM'")
            print("   - 'LLM response received'")
            print("   - 'JSON parsed successfully'")
    else:
        print(f"❌ Agents failed: {results.get('error', 'Unknown error')}")
else:
    print(f"❌ Failed to get results: {response.status_code}")

print("\n" + "=" * 60)
print("Test complete!")