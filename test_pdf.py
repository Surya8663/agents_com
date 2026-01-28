# test_with_proper_timeouts.py
import requests
import time
import json

print("🚀 Testing with Proper Timeouts")
print("=" * 60)

# Create a simple PDF
from reportlab.pdfgen import canvas
c = canvas.Canvas("timeout_test.pdf")
c.drawString(100, 700, "Invoice #: TEST-TIMEOUT-001")
c.drawString(100, 680, "Date: March 1, 2024")
c.drawString(100, 660, "Customer: Timeout Test Inc.")
c.drawString(100, 640, "Amount: $250.00")
c.drawString(100, 620, "Status: Paid")
c.save()

print("1. Uploading document...")
with open("timeout_test.pdf", "rb") as f:
    files = {'file': ('timeout_test.pdf', f, 'application/pdf')}
    response = requests.post("http://localhost:8000/ingest/upload", files=files, timeout=60)
    
if response.status_code == 200:
    doc_id = response.json().get('document_id')
    print(f"✅ Document uploaded: {doc_id}")
else:
    print(f"❌ Upload failed: {response.status_code}")
    exit()

print("\n2. Starting Phase 2 (Layout) - This may take 60+ seconds...")
try:
    # Start layout with NO response timeout, but connection timeout
    response = requests.post(f"http://localhost:8000/layout/analyze/{doc_id}", timeout=(10, None))
    print(f"   Response: {response.json().get('message', 'Started')}")
except requests.exceptions.ReadTimeout:
    print("   ⏳ Layout started (taking time) - this is normal for YOLO")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Wait longer for YOLO
print("   Waiting 90 seconds for YOLO processing...")
time.sleep(90)

print("\n3. Checking Phase 2 status...")
try:
    response = requests.get(f"http://localhost:8000/layout/status/{doc_id}", timeout=10)
    if response.status_code == 200:
        status = response.json()
        print(f"   Layout status: {status.get('status', 'unknown')}")
        if status.get('status') == 'completed':
            print("   ✅ Phase 2 completed!")
        else:
            print(f"   ⏳ Still processing: {status}")
except Exception as e:
    print(f"   ❌ Status check failed: {e}")

# Continue with Phase 3 if Phase 2 completed
print("\n4. Starting Phase 3 (OCR)...")
try:
    response = requests.post(f"http://localhost:8000/ocr/process/{doc_id}", timeout=30)
    print(f"   Response: {response.json().get('message', 'Started')}")
except Exception as e:
    print(f"   ❌ OCR failed: {e}")

# Wait for OCR
print("   Waiting 30 seconds for OCR...")
time.sleep(30)

print("\n5. Starting Phase 4 (Agents)...")
try:
    response = requests.post(f"http://localhost:8000/agents/run/{doc_id}", timeout=30)
    print(f"   Response: {response.json().get('message', 'Started')}")
    print(f"   Agents: {response.json().get('agents', [])}")
except Exception as e:
    print(f"   ❌ Agents failed: {e}")

# Wait for LLM processing
print("\n6. Waiting 60 seconds for LLM agents...")
time.sleep(60)

print("\n7. Checking final results...")
try:
    response = requests.get(f"http://localhost:8000/agents/result/{doc_id}", timeout=10)
    if response.status_code == 200:
        results = response.json()
        
        print(f"   Overall status: {results.get('status', 'unknown')}")
        print(f"   Agents executed: {results.get('agents_executed', [])}")
        
        if results.get('status') == 'completed':
            # Check Text Agent
            text_analysis = results.get('text_analysis', {})
            print(f"\n   Text Agent Results:")
            print(f"     Document type: {text_analysis.get('document_type', 'unknown')}")
            
            key_value_pairs = text_analysis.get('key_value_pairs', {})
            if key_value_pairs:
                print(f"     ✅ REAL DATA EXTRACTED:")
                for key, value in key_value_pairs.items():
                    print(f"       • {key}: {value}")
            else:
                print("     ⚠️ No fields extracted")
                
        else:
            print(f"   ❌ Agents failed: {results.get('error', 'Unknown')}")
    else:
        print(f"   ❌ Failed to get results: {response.status_code}")
        
except Exception as e:
    print(f"   ❌ Results check failed: {e}")

print("\n" + "=" * 60)
print("Test complete! Check server logs for:")
print("   - YOLO inference messages")
print("   - 'Text Agent calling LLM'")
print("   - 'LLM response received'")