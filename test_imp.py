# test_diagnosis.py
"""
Diagnostic script to find EXACTLY where the issue is.
Run this after uploading a document but before running agents.
"""
import json
import time
import uuid
import requests
import sys
from pathlib import Path
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def print_header(text: str):
    print("\n" + "="*60)
    print(f"🔍 {text}")
    print("="*60)

def print_step(text: str):
    print(f"\n👉 {text}")

def print_success(text: str):
    print(f"✅ {text}")

def print_error(text: str):
    print(f"❌ {text}")

def print_warning(text: str):
    print(f"⚠️  {text}")

def print_info(text: str):
    print(f"ℹ️  {text}")

def test_endpoint(method: str, endpoint: str, **kwargs):
    """Test an API endpoint."""
    url = f"{BASE_URL}{endpoint}"
    print_step(f"Testing {method} {endpoint}")
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, **kwargs)
        elif method.upper() == "POST":
            response = requests.post(url, **kwargs)
        else:
            print_error(f"Unsupported method: {method}")
            return None
        
        print_info(f"Status: {response.status_code}")
        if response.status_code != 200:
            print_error(f"Response: {response.text[:200]}")
        else:
            print_success("Request successful")
        
        return response
        
    except Exception as e:
        print_error(f"Request failed: {e}")
        return None

def check_file_system(document_id: str):
    """Check what files actually exist on disk."""
    print_header("FILE SYSTEM CHECK")
    
    # Assuming default data directory
    data_dir = Path("data") / "documents" / document_id
    
    if not data_dir.exists():
        print_error(f"Document directory doesn't exist: {data_dir}")
        return False
    
    print_success(f"Document directory exists: {data_dir}")
    
    # Check all subdirectories
    for subdir in ["pages", "layout", "ocr", "agents"]:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob("*"))
            json_files = list(subdir_path.glob("*.json"))
            print_success(f"{subdir}/: {len(files)} total files, {len(json_files)} JSON files")
            
            # Show JSON file names
            if json_files:
                print_info(f"  JSON files: {[f.name for f in json_files[:5]]}")
                if len(json_files) > 5:
                    print_info(f"  ... and {len(json_files) - 5} more")
            
            # Check file contents for first JSON file
            if json_files and subdir in ["layout", "ocr"]:
                sample_file = json_files[0]
                try:
                    with open(sample_file, "r") as f:
                        content = json.load(f)
                        print_info(f"  Sample file '{sample_file.name}': {len(content)} keys")
                        if isinstance(content, dict):
                            print_info(f"    Keys: {list(content.keys())[:5]}")
                except Exception as e:
                    print_error(f"  Could not read {sample_file.name}: {e}")
        else:
            print_error(f"{subdir}/: Directory doesn't exist")
    
    return True

def test_phase_1(document_id: str):
    """Test Phase 1: Document Upload."""
    print_header("PHASE 1: DOCUMENT UPLOAD TEST")
    
    # This assumes document is already uploaded
    response = test_endpoint("GET", f"/ingest/status/{document_id}")
    if not response:
        return False
    
    if response.status_code == 200:
        data = response.json()
        print_info(f"Status: {data.get('status', 'unknown')}")
        print_info(f"Pages: {data.get('page_count', 0)}")
        return data.get('status') == 'completed'
    
    return False

def test_phase_2(document_id: str):
    """Test Phase 2: Layout Analysis."""
    print_header("PHASE 2: LAYOUT ANALYSIS TEST")
    
    # Check layout status
    response = test_endpoint("GET", f"/layout/status/{document_id}")
    if not response:
        return False
    
    if response.status_code == 200:
        data = response.json()
        layout_status = data.get('status', 'unknown')
        print_info(f"Layout status: {layout_status}")
        
        if layout_status == 'completed':
            # Try to get layout results
            response2 = test_endpoint("GET", f"/layout/results/{document_id}")
            if response2 and response2.status_code == 200:
                results = response2.json()
                pages = results.get('pages', [])
                print_success(f"Layout has {len(pages)} pages")
                if pages:
                    detections = pages[0].get('detections', [])
                    print_info(f"First page has {len(detections)} detections")
                return True
        elif layout_status == 'processing':
            print_warning("Layout is still processing...")
            # Wait and check again
            print_info("Waiting 10 seconds for layout to complete...")
            time.sleep(10)
            return test_phase_2(document_id)  # Recursive check
        else:
            # Try to trigger layout analysis
            print_warning("Layout not started or failed, trying to start...")
            response3 = test_endpoint("POST", f"/layout/analyze/{document_id}")
            if response3 and response3.status_code == 200:
                print_info("Layout analysis started, waiting...")
                time.sleep(5)
                return test_phase_2(document_id)
    
    return False

def test_phase_3(document_id: str):
    """Test Phase 3: OCR Processing."""
    print_header("PHASE 3: OCR PROCESSING TEST")
    
    # Check OCR status
    response = test_endpoint("GET", f"/ocr/status/{document_id}")
    if not response:
        return False
    
    if response.status_code == 200:
        data = response.json()
        ocr_status = data.get('status', 'unknown')
        print_info(f"OCR status: {ocr_status}")
        
        if ocr_status == 'completed':
            # Try to get OCR results
            response2 = test_endpoint("GET", f"/ocr/results/{document_id}")
            if response2 and response2.status_code == 200:
                results = response2.json()
                pages = results.get('pages', [])
                print_success(f"OCR has {len(pages)} pages")
                if pages:
                    regions = pages[0].get('regions', [])
                    print_info(f"First page has {len(regions)} regions")
                    if regions:
                        has_text = any(r.get('ocr_text') for r in regions)
                        print_info(f"First page has OCR text: {has_text}")
                return True
        elif ocr_status == 'processing':
            print_warning("OCR is still processing...")
            # Wait and check again
            print_info("Waiting 10 seconds for OCR to complete...")
            time.sleep(10)
            return test_phase_3(document_id)  # Recursive check
        else:
            # Try to trigger OCR
            print_warning("OCR not started or failed, trying to start...")
            response3 = test_endpoint("POST", f"/ocr/process/{document_id}")
            if response3 and response3.status_code == 200:
                print_info("OCR started, waiting...")
                time.sleep(5)
                return test_phase_3(document_id)
    
    return False

def test_phase_4(document_id: str):
    """Test Phase 4: Agent Pipeline."""
    print_header("PHASE 4: AGENT PIPELINE TEST")
    
    # First check prerequisites
    response = test_endpoint("GET", f"/agents/prerequisites/{document_id}")
    if not response:
        return False
    
    if response.status_code == 200:
        data = response.json()
        prerequisites = data.get('prerequisites', {})
        print_info(f"Layout ready: {prerequisites.get('layout_ready', False)}")
        print_info(f"OCR ready: {prerequisites.get('ocr_ready', False)}")
        print_info(f"Has any data: {prerequisites.get('has_any_data', False)}")
        
        if prerequisites.get('has_any_data'):
            # Try to run agents
            print_info("Attempting to run agents...")
            response2 = test_endpoint("POST", f"/agents/run/{document_id}")
            if response2:
                if response2.status_code == 200:
                    result = response2.json()
                    print_success(f"Agents started: {result.get('status', 'unknown')}")
                    print_info(f"Note: {result.get('note', '')}")
                    
                    # Wait and check status
                    print_info("Waiting 5 seconds for agent status...")
                    time.sleep(5)
                    
                    response3 = test_endpoint("GET", f"/agents/status/{document_id}")
                    if response3 and response3.status_code == 200:
                        status_data = response3.json()
                        print_info(f"Agent status: {status_data.get('status', 'unknown')}")
                        return True
                elif response2.status_code == 400:
                    error_text = response2.text
                    print_error(f"Agent start failed: {error_text}")
                    return False
        else:
            print_error("No data available for agents (both layout and OCR failed)")
    
    return False

def test_ocr_engine():
    """Test if OCR engine is working."""
    print_header("OCR ENGINE TEST")
    
    response = test_endpoint("GET", "/ocr/engine/info")
    if not response:
        return False
    
    if response.status_code == 200:
        data = response.json()
        engine = data.get('engine', 'unknown')
        mode = data.get('mode', 'unknown')
        using_real_ocr = data.get('using_real_ocr', False)
        
        print_info(f"Engine: {engine}")
        print_info(f"Mode: {mode}")
        print_info(f"Using real OCR: {using_real_ocr}")
        
        if not using_real_ocr or mode != 'real':
            print_error("OCR engine is NOT using real OCR! This is the problem!")
            print_error("Check app/ocr/engine.py - it should be using REAL EasyOCR")
            return False
        else:
            print_success("OCR engine is configured for REAL OCR")
            return True
    
    return False

def test_layout_model():
    """Test if layout model is working."""
    print_header("LAYOUT MODEL TEST")
    
    response = test_endpoint("GET", "/layout/model/info")
    if not response:
        return False
    
    if response.status_code == 200:
        data = response.json()
        status = data.get('status', 'unknown')
        print_info(f"Model status: {status}")
        
        if status == 'loaded':
            print_success("Layout model is loaded")
            return True
        else:
            print_error(f"Layout model not loaded: {data.get('error', 'Unknown error')}")
            return False
    
    return False

def wait_for_background_tasks(document_id: str, timeout: int = 60):
    """Wait for background tasks to complete."""
    print_header(f"WAITING FOR BACKGROUND TASKS (max {timeout}s)")
    
    start_time = time.time()
    last_layout_status = None
    last_ocr_status = None
    
    while time.time() - start_time < timeout:
        # Check layout status
        layout_response = requests.get(f"{BASE_URL}/layout/status/{document_id}")
        if layout_response.status_code == 200:
            layout_data = layout_response.json()
            current_layout_status = layout_data.get('status')
            if current_layout_status != last_layout_status:
                print_info(f"Layout status: {current_layout_status}")
                last_layout_status = current_layout_status
        
        # Check OCR status
        ocr_response = requests.get(f"{BASE_URL}/ocr/status/{document_id}")
        if ocr_response.status_code == 200:
            ocr_data = ocr_response.json()
            current_ocr_status = ocr_data.get('status')
            if current_ocr_status != last_ocr_status:
                print_info(f"OCR status: {current_ocr_status}")
                last_ocr_status = current_ocr_status
        
        # Check if both are completed
        if (last_layout_status == 'completed' and 
            last_ocr_status == 'completed'):
            print_success("Both layout and OCR completed!")
            return True
        
        # Wait before checking again
        time.sleep(2)
    
    print_warning(f"Timeout after {timeout} seconds")
    print_info(f"Final layout status: {last_layout_status}")
    print_info(f"Final OCR status: {last_ocr_status}")
    return False

def main():
    """Main diagnostic function."""
    print_header("DOCUMENT INTELLIGENCE SYSTEM DIAGNOSTIC")
    
    # Ask for document ID
    if len(sys.argv) > 1:
        document_id = sys.argv[1]
    else:
        document_id = input("Enter document ID: ").strip()
    
    if not document_id:
        print_error("No document ID provided")
        return
    
    # Validate UUID format
    try:
        uuid.UUID(document_id)
    except ValueError:
        print_error(f"Invalid UUID format: {document_id}")
        return
    
    print_success(f"Testing document: {document_id}")
    
    # Run diagnostic tests
    test_results = {}
    
    # Test 0: File system check
    test_results['filesystem'] = check_file_system(document_id)
    
    # Test 1: OCR engine
    test_results['ocr_engine'] = test_ocr_engine()
    
    # Test 2: Layout model
    test_results['layout_model'] = test_layout_model()
    
    # Test 3: Wait for background tasks
    test_results['background_tasks'] = wait_for_background_tasks(document_id, timeout=30)
    
    # Test 4: Phase 1
    test_results['phase1'] = test_phase_1(document_id)
    
    # Test 5: Phase 2
    test_results['phase2'] = test_phase_2(document_id)
    
    # Test 6: Phase 3
    test_results['phase3'] = test_phase_3(document_id)
    
    # Test 7: Phase 4
    test_results['phase4'] = test_phase_4(document_id)
    
    # Summary
    print_header("DIAGNOSTIC SUMMARY")
    
    for test_name, result in test_results.items():
        if result:
            print_success(f"{test_name}: PASS")
        else:
            print_error(f"{test_name}: FAIL")
    
    # Determine the root cause
    print_header("ROOT CAUSE ANALYSIS")
    
    if not test_results.get('ocr_engine'):
        print_error("❌ ROOT CAUSE: OCR engine is NOT using real EasyOCR")
        print_error("   Fix: Check app/ocr/engine.py - should use REAL EasyOCR, not mock")
        print_error("   Run: pip install easyocr")
        
    elif not test_results.get('layout_model'):
        print_error("❌ ROOT CAUSE: Layout model failed to load")
        print_error("   Fix: Check YOLO model files in app/vision/")
        
    elif not test_results.get('phase2'):
        print_error("❌ ROOT CAUSE: Layout analysis failed")
        print_error("   Check: app/api/layout.py - background task might be failing")
        
    elif not test_results.get('phase3'):
        print_error("❌ ROOT CAUSE: OCR processing failed")
        print_error("   Check: app/api/ocr.py - OCR might be failing silently")
        
    elif not test_results.get('phase4'):
        print_error("❌ ROOT CAUSE: Agent pipeline failed")
        print_error("   Check: app/api/agents.py - prerequisites not met")
        
    else:
        print_success("✅ ALL TESTS PASSED!")
        print_success("System is working correctly!")
    
    print_header("NEXT STEPS")
    
    # Check specific directories
    data_dir = Path("data") / "documents" / document_id
    layout_dir = data_dir / "layout"
    ocr_dir = data_dir / "ocr"
    
    if layout_dir.exists():
        layout_files = list(layout_dir.glob("*.json"))
        print_info(f"Layout files: {len(layout_files)}")
        if layout_files:
            print_info(f"  Files: {[f.name for f in layout_files]}")
    
    if ocr_dir.exists():
        ocr_files = list(ocr_dir.glob("*.json"))
        print_info(f"OCR files: {len(ocr_files)}")
        if ocr_files:
            print_info(f"  Files: {[f.name for f in ocr_files]}")

if __name__ == "__main__":
    main()