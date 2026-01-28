# debug_full_pipeline.py
import requests
import json
import os
import sys
from pathlib import Path

def check_phase1_upload():
    """Test Phase 1: Upload a document."""
    print("\n📤 PHASE 1: Document Upload")
    print("-" * 40)
    
    # Create a simple test PDF
    from reportlab.pdfgen import canvas
    from io import BytesIO
    
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, "INVOICE")
    c.drawString(100, 680, "Invoice #: INV-2024-001")
    c.drawString(100, 660, "Date: January 27, 2024")
    c.drawString(100, 640, "Customer: Tech Corp Inc.")
    c.drawString(100, 620, "Item: Advanced AI Software License")
    c.drawString(100, 600, "Quantity: 1")
    c.drawString(100, 580, "Price: $1,499.00")
    c.drawString(100, 560, "Tax: $299.80")
    c.drawString(100, 540, "Total: $1,798.80")
    c.drawString(100, 520, "Payment Due: February 10, 2024")
    c.save()
    
    buffer.seek(0)
    files = {'file': ('debug_invoice.pdf', buffer, 'application/pdf')}
    
    try:
        response = requests.post("http://localhost:8000/ingest/upload", files=files, timeout=30)
        if response.status_code == 200:
            doc_info = response.json()
            doc_id = doc_info.get('document_id')
            print(f"✅ Document uploaded: {doc_id}")
            print(f"   Pages: {doc_info.get('pages', 0)}")
            print(f"   Message: {doc_info.get('message', '')}")
            return doc_id
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None

def check_phase2_layout(doc_id):
    """Check Phase 2: Layout Analysis."""
    print("\n📐 PHASE 2: Layout Analysis")
    print("-" * 40)
    
    # Start layout analysis
    try:
        print("Starting layout analysis...")
        response = requests.post(f"http://localhost:8000/layout/analyze/{doc_id}", timeout=30)
        if response.status_code in [200, 202]:
            print(f"✅ Layout started: {response.json().get('message', '')}")
            
            # Wait and check results
            import time
            for i in range(10):
                time.sleep(3)
                status_response = requests.get(f"http://localhost:8000/layout/status/{doc_id}", timeout=10)
                if status_response.status_code == 200:
                    status = status_response.json()
                    current_status = status.get('status', 'unknown')
                    print(f"   Check {i+1}: Status = {current_status}")
                    
                    if current_status == 'completed':
                        # Get layout results
                        results_response = requests.get(f"http://localhost:8000/layout/results/{doc_id}", timeout=10)
                        if results_response.status_code == 200:
                            layout_data = results_response.json()
                            print(f"✅ Layout analysis completed")
                            
                            # Check data structure
                            if "pages" in layout_data:
                                pages = layout_data["pages"]
                                print(f"   Pages analyzed: {len(pages)}")
                                total_detections = sum(len(page.get("detections", [])) for page in pages)
                                print(f"   Total detections: {total_detections}")
                                
                                if total_detections > 0:
                                    # Show first detection
                                    for page in pages[:1]:
                                        detections = page.get("detections", [])
                                        if detections:
                                            first_det = detections[0]
                                            print(f"   Sample detection:")
                                            print(f"     Type: {first_det.get('type', 'unknown')}")
                                            print(f"     BBox: {first_det.get('bbox', {})}")
                                            print(f"     Confidence: {first_det.get('confidence', 0.0)}")
                                    return True, layout_data
                                else:
                                    print("⚠️ No detections found in layout")
                                    return False, None
                            else:
                                print("❌ No 'pages' key in layout data")
                                return False, None
                        break
                    elif current_status == 'failed':
                        print(f"❌ Layout failed: {status.get('error', 'Unknown error')}")
                        break
            else:
                print("⚠️ Layout timed out")
                return False, None
        else:
            print(f"❌ Layout start failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False, None
            
    except Exception as e:
        print(f"❌ Layout error: {e}")
        return False, None

def check_phase3_ocr(doc_id):
    """Check Phase 3: OCR Processing."""
    print("\n🔤 PHASE 3: OCR Processing")
    print("-" * 40)
    
    # Start OCR processing
    try:
        print("Starting OCR processing...")
        response = requests.post(f"http://localhost:8000/ocr/process/{doc_id}", timeout=30)
        if response.status_code in [200, 202]:
            print(f"✅ OCR started: {response.json().get('message', '')}")
            
            # Wait and check results
            import time
            for i in range(10):
                time.sleep(3)
                status_response = requests.get(f"http://localhost:8000/ocr/status/{doc_id}", timeout=10)
                if status_response.status_code == 200:
                    status = status_response.json()
                    current_status = status.get('status', 'unknown')
                    print(f"   Check {i+1}: Status = {current_status}")
                    
                    if current_status == 'completed':
                        # Get OCR results
                        results_response = requests.get(f"http://localhost:8000/ocr/results/{doc_id}", timeout=10)
                        if results_response.status_code == 200:
                            ocr_data = results_response.json()
                            print(f"✅ OCR processing completed")
                            
                            # Check data structure
                            if "pages" in ocr_data:
                                pages = ocr_data["pages"]
                                print(f"   Pages processed: {len(pages)}")
                                total_regions = sum(len(page.get("regions", [])) for page in pages)
                                print(f"   Total regions: {total_regions}")
                                
                                # Count regions with text
                                text_regions = 0
                                all_text = ""
                                for page in pages:
                                    for region in page.get("regions", []):
                                        text = region.get("ocr_text", "")
                                        if text and len(text.strip()) > 0:
                                            text_regions += 1
                                            all_text += text + " "
                                
                                print(f"   Regions with text: {text_regions}")
                                
                                if text_regions > 0:
                                    print(f"   Sample text: {all_text[:200]}...")
                                    return True, ocr_data
                                else:
                                    print("⚠️ No text found in OCR results")
                                    return False, None
                            else:
                                print("❌ No 'pages' key in OCR data")
                                return False, None
                        break
                    elif current_status == 'failed':
                        print(f"❌ OCR failed: {status.get('error', 'Unknown error')}")
                        break
            else:
                print("⚠️ OCR timed out")
                return False, None
        else:
            print(f"❌ OCR start failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False, None
            
    except Exception as e:
        print(f"❌ OCR error: {e}")
        return False, None

def check_phase4_agents(doc_id, layout_data, ocr_data):
    """Check Phase 4: Multi-Modal Agents."""
    print("\n🧠 PHASE 4: Multi-Modal Agents")
    print("-" * 40)
    
    # First, check if agents can run with current data
    print("1. Checking prerequisites...")
    
    if layout_data is None:
        print("❌ No layout data from Phase 2")
        return False
    
    if ocr_data is None:
        print("❌ No OCR data from Phase 3")
        return False
    
    # Check file structure on disk
    from app.config.settings import settings
    doc_dir = settings.DOCUMENTS_DIR / doc_id
    
    print(f"2. Document directory: {doc_dir}")
    
    # Check layout files
    layout_dir = doc_dir / "layout"
    if layout_dir.exists():
        layout_files = list(layout_dir.glob("*.json"))
        print(f"   Layout files: {len(layout_files)}")
    else:
        print("⚠️ Layout directory not found")
    
    # Check OCR files
    ocr_dir = doc_dir / "ocr"
    if ocr_dir.exists():
        ocr_files = list(ocr_dir.glob("*.json"))
        print(f"   OCR files: {len(ocr_files)}")
    else:
        print("⚠️ OCR directory not found")
    
    # Start agent pipeline
    try:
        print("\n3. Starting agent pipeline...")
        response = requests.post(f"http://localhost:8000/agents/run/{doc_id}?force_rerun=true", timeout=30)
        if response.status_code in [200, 202]:
            agent_response = response.json()
            print(f"✅ Agents started: {agent_response.get('message', '')}")
            print(f"   Agents: {agent_response.get('agents', [])}")
            
            # Monitor progress
            import time
            print("\n4. Monitoring agent progress (this will take 30-60 seconds for LLM)...")
            
            for i in range(1, 25):  # 24 checks, 5 seconds apart = 120 seconds total
                time.sleep(5)
                
                try:
                    status_response = requests.get(f"http://localhost:8000/agents/status/{doc_id}", timeout=10)
                    if status_response.status_code == 200:
                        status = status_response.json()
                        current_status = status.get('status', 'unknown')
                        
                        print(f"   Check {i} ({i*5}s): Status = {current_status}")
                        
                        if current_status == 'completed':
                            print("\n✅ Agents completed!")
                            
                            # Get final results
                            results_response = requests.get(f"http://localhost:8000/agents/result/{doc_id}", timeout=10)
                            if results_response.status_code == 200:
                                results = results_response.json()
                                print("\n📊 FINAL AGENT RESULTS:")
                                
                                # Check key sections
                                print(f"   Agents executed: {results.get('agents_executed', [])}")
                                
                                # Check Text Agent output
                                text_analysis = results.get('text_analysis', {})
                                print(f"\n   Text Agent Results:")
                                print(f"     Document type: {text_analysis.get('document_type', 'unknown')}")
                                print(f"     Semantic confidence: {text_analysis.get('semantic_confidence', 0.0):.2f}")
                                
                                key_value_pairs = text_analysis.get('key_value_pairs', {})
                                print(f"     Key-value pairs extracted: {len(key_value_pairs)}")
                                if key_value_pairs:
                                    print("     Extracted fields:")
                                    for key, value in list(key_value_pairs.items())[:5]:
                                        print(f"       • {key}: {value}")
                                
                                # Check Fusion Agent output
                                fused_doc = results.get('fused_document', {})
                                print(f"\n   Fusion Agent Results:")
                                print(f"     Fusion confidence: {fused_doc.get('fusion_confidence', 0.0):.2f}")
                                
                                fused_extractions = fused_doc.get('fused_extractions', {})
                                print(f"     Fused extractions: {len(fused_extractions)}")
                                
                                # Check Validation Agent output
                                validation = results.get('validation_result', {})
                                print(f"\n   Validation Agent Results:")
                                print(f"     Overall confidence: {validation.get('overall_confidence', 0.0):.2f}")
                                print(f"     Passed checks: {len(validation.get('validation_passed', []))}")
                                print(f"     Failed checks: {len(validation.get('validation_failed', []))}")
                                
                                # Check Final Output
                                final_output = results.get('final_output', {})
                                print(f"\n   Final Output:")
                                extracted_fields = final_output.get('extracted_fields', {})
                                print(f"     Extracted fields in final output: {len(extracted_fields)}")
                                
                                if extracted_fields:
                                    print("🎉 REAL DATA EXTRACTED SUCCESSFULLY!")
                                    for key, value_info in list(extracted_fields.items())[:10]:
                                        if isinstance(value_info, dict):
                                            value = value_info.get('value', 'N/A')
                                            confidence = value_info.get('confidence', 'N/A')
                                            print(f"       • {key}: {value} (confidence: {confidence})")
                                        else:
                                            print(f"       • {key}: {value_info}")
                                else:
                                    print("⚠️ No fields in final output - checking why...")
                                    
                                    # Check errors
                                    errors = results.get('errors', [])
                                    if errors:
                                        print("     Errors found:")
                                        for error in errors:
                                            print(f"       - {error}")
                                    
                                    # Check if agents had errors
                                    for agent_name in ['text_analysis', 'fusion_agent', 'validation_result']:
                                        agent_data = results.get(agent_name, {})
                                        if 'error' in agent_data:
                                            print(f"     {agent_name} error: {agent_data['error']}")
                            
                            return True
                        elif current_status == 'failed':
                            print(f"\n❌ Agents failed: {status.get('error', 'Unknown error')}")
                            return False
                except Exception as e:
                    print(f"   Check {i}: Error - {e}")
            
            print("\n⏱️  Agent pipeline timed out after 120 seconds")
            print("   Checking what happened...")
            
            # Even if timed out, check if any results exist
            try:
                results_response = requests.get(f"http://localhost:8000/agents/result/{doc_id}", timeout=10)
                if results_response.status_code == 200:
                    results = results_response.json()
                    print(f"\n📄 Partial results found:")
                    print(f"   Status in results: {results.get('status', 'unknown')}")
                    
                    # Check if agents executed
                    agents_executed = results.get('agents_executed', [])
                    print(f"   Agents executed: {len(agents_executed)}")
                    
                    if len(agents_executed) < 4:
                        print(f"⚠️ Only {len(agents_executed)}/4 agents executed")
                        print(f"   Missing: {set(['vision_agent', 'text_agent', 'fusion_agent', 'validation_agent']) - set(agents_executed)}")
            except:
                pass
            
            return False
        else:
            print(f"❌ Agents start failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Agents error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_server_health():
    """Check if server is running."""
    print("🏥 Checking Server Health")
    print("-" * 40)
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Server is running")
            print(f"   Service: {health.get('service', 'unknown')}")
            print(f"   Version: {health.get('version', 'unknown')}")
            return True
        else:
            print(f"❌ Server returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        return False

def check_llm_config():
    """Check LLM configuration."""
    print("\n🤖 Checking LLM Configuration")
    print("-" * 40)
    
    try:
        response = requests.get("http://localhost:8000/agents/llm/info", timeout=10)
        if response.status_code == 200:
            llm_info = response.json()
            print(f"✅ LLM configured")
            print(f"   Model: {llm_info.get('model', 'unknown')}")
            print(f"   Provider: {llm_info.get('llm_provider', 'unknown')}")
            print(f"   Status: {llm_info.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ LLM info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ LLM check error: {e}")
        return False

def main():
    print("🔧 COMPLETE PIPELINE DEBUGGING")
    print("=" * 60)
    
    # Check server health
    if not check_server_health():
        print("\n❌ Server is not running. Start it with: uvicorn app.main:app --reload")
        return
    
    # Check LLM config
    if not check_llm_config():
        print("\n⚠️ LLM may not be properly configured")
    
    # Upload test document
    doc_id = check_phase1_upload()
    if not doc_id:
        print("\n❌ Cannot proceed without document")
        return
    
    print(f"\n📋 Using Document ID: {doc_id}")
    
    # Run Phase 2
    layout_success, layout_data = check_phase2_layout(doc_id)
    
    # Run Phase 3
    ocr_success, ocr_data = check_phase3_ocr(doc_id)
    
    # Run Phase 4
    if layout_success and ocr_success:
        print("\n" + "=" * 60)
        print("🚀 ALL PREREQUISITES MET - RUNNING AGENTS")
        print("=" * 60)
        
        agent_success = check_phase4_agents(doc_id, layout_data, ocr_data)
        
        if agent_success:
            print("\n🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
        else:
            print("\n❌ AGENTS FAILED - See details above")
    else:
        print("\n❌ Cannot run agents - Phases 2 or 3 failed")
        print(f"   Phase 2 (Layout): {'✅' if layout_success else '❌'}")
        print(f"   Phase 3 (OCR): {'✅' if ocr_success else '❌'}")
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("-" * 40)
    print(f"Document ID: {doc_id}")
    print(f"Phase 2 Layout: {'✅' if layout_success else '❌'}")
    print(f"Phase 3 OCR: {'✅' if ocr_success else '❌'}")
    print(f"Phase 4 Agents: {'✅' if 'agent_success' in locals() and agent_success else '❌' if 'agent_success' in locals() else 'N/A'}")
    
    print("\n💡 NEXT STEPS:")
    if 'agent_success' in locals() and not agent_success:
        print("1. Check server logs for LLM call errors")
        print("2. Verify Ollama is running: ollama serve")
        print("3. Check if agents have proper data from Phases 2 & 3")
    elif layout_success and ocr_success and 'agent_success' not in locals():
        print("Run agents manually: curl -X POST http://localhost:8000/agents/run/{doc_id}?force_rerun=true")

if __name__ == "__main__":
    main()