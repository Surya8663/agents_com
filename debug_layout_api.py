# test_full_api_after_fix.py
import requests
import time
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_complete_workflow():
    """Test the complete Phase 2 workflow after the fix."""
    print("Testing Complete Phase 2 Workflow After Fix")
    print("=" * 70)
    
    # Check server
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✓ Server is running")
        else:
            print("✗ Server not responding properly")
            return
    except:
        print("✗ Server not running")
        return
    
    # Upload test PDF
    print("\n1. Uploading test PDF...")
    pdf_path = "better_test_document.pdf"
    
    if not Path(pdf_path).exists():
        print(f"✗ PDF not found: {pdf_path}")
        return
    
    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path, f, 'application/pdf')}
        response = requests.post(f"{BASE_URL}/ingest/upload", files=files)
    
    if response.status_code != 200:
        print(f"✗ Upload failed: {response.text}")
        return
    
    upload_result = response.json()
    document_id = upload_result['document_id']
    print(f"✓ Upload successful")
    print(f"  Document ID: {document_id}")
    print(f"  Pages: {upload_result['total_pages']}")
    
    # Wait for Phase 1 processing
    print("\n2. Waiting for Phase 1 processing...")
    time.sleep(3)
    
    # Start Phase 2 layout analysis
    print(f"\n3. Starting layout analysis...")
    response = requests.post(f"{BASE_URL}/layout/analyze/{document_id}")
    
    if response.status_code != 200:
        print(f"✗ Failed to start analysis: {response.text}")
        return
    
    start_result = response.json()
    print(f"✓ {start_result['message']}")
    
    # Monitor progress
    print(f"\n4. Monitoring progress...")
    max_checks = 15
    for i in range(max_checks):
        time.sleep(2)
        
        response = requests.get(f"{BASE_URL}/layout/status/{document_id}")
        if response.status_code == 200:
            status = response.json()
            current_status = status.get('status', 'unknown')
            print(f"  Check {i+1}: Status = {current_status}")
            
            if current_status == 'completed':
                print("  ✓ Analysis completed!")
                break
            elif current_status == 'failed':
                print("  ✗ Analysis failed")
                if 'error' in status:
                    print(f"  Error: {status['error']}")
                return
        else:
            print(f"  ✗ Status check failed: {response.text}")
    
    # Get results
    print(f"\n5. Getting layout results...")
    response = requests.get(f"{BASE_URL}/layout/results/{document_id}")
    
    if response.status_code == 200:
        results = response.json()
        
        if 'pages' in results:
            pages = results['pages']
            print(f"✓ Got results for {len(pages)} pages")
            
            # Analyze results
            total_detections = 0
            detection_types = {}
            
            for i, page in enumerate(pages, 1):
                detections = page.get('detections', [])
                total_detections += len(detections)
                
                for det in detections:
                    det_type = det.get('type', 'unknown')
                    detection_types[det_type] = detection_types.get(det_type, 0) + 1
            
            print(f"\n  Summary:")
            print(f"  Total detections across all pages: {total_detections}")
            print(f"  Detection types:")
            for det_type, count in detection_types.items():
                print(f"    {det_type}: {count}")
            
            # Save results to file for inspection
            output_file = f"layout_results_{document_id}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n  Full results saved to: {output_file}")
            
            return True
        else:
            print(f"✗ Unexpected response format: {results}")
            return False
    else:
        print(f"✗ Failed to get results: {response.text}")
        return False

if __name__ == "__main__":
    success = test_complete_workflow()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ PHASE 2 WORKFLOW TEST PASSED!")
    else:
        print("❌ PHASE 2 WORKFLOW TEST FAILED")
    print("=" * 70)