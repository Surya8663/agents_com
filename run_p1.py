
"""
FINAL COMPREHENSIVE TEST - Tests all 4 phases end-to-end
with real verification of outputs.
STRICTLY NO MOCK DATA - ALL REAL EXECUTION
"""
import asyncio
import requests
import json
import time
import sys
from pathlib import Path
import uuid
import os

BASE_URL = "http://localhost:8000"
TEST_PDF = "final_test_document.pdf"

class CompleteTester:
    """Comprehensive tester for all 4 phases - NO MOCKING."""
    
    def __init__(self):
        self.document_id = None
        self.results = {
            "phase1": {"status": "not_started"},
            "phase2": {"status": "not_started"},
            "phase3": {"status": "not_started"},
            "phase4": {"status": "not_started"}
        }
    
    def create_test_document(self):
        """Create a comprehensive test PDF with REAL content."""
        print("\n📝 Creating comprehensive test document...")
        
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            
            doc = SimpleDocTemplate(TEST_PDF, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title = Paragraph("COMPREHENSIVE TEST DOCUMENT", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 0.2*inch))
            
            # Header
            header = Paragraph("Multi-Modal Document Intelligence System - Final Test", styles['Heading1'])
            story.append(header)
            story.append(Spacer(1, 0.2*inch))
            
            # Document info
            info = Paragraph(
                "<b>Document ID:</b> TEST-DOC-001<br/>"
                "<b>Date:</b> 2024-01-15<br/>"
                "<b>Purpose:</b> End-to-end testing of all 4 phases<br/>"
                "<b>Test Type:</b> Comprehensive validation",
                styles['Normal']
            )
            story.append(info)
            story.append(Spacer(1, 0.3*inch))
            
            # Invoice table with CLEAR TEXT
            data = [
                ['Item Description', 'Qty', 'Unit Price', 'Total'],
                ['Advanced AI Software License', '2', '$1,499.00', '$2,998.00'],
                ['Technical Support (Annual)', '1', '$500.00', '$500.00'],
                ['Training Sessions', '5', '$200.00', '$1,000.00'],
                ['Custom Integration', '1', '$2,500.00', '$2,500.00'],
                ['', '', '<b>Subtotal:</b>', '<b>$6,998.00</b>'],
                ['', '', 'Tax (8.5%):', '$594.83'],
                ['', '', '<b>GRAND TOTAL:</b>', '<b>$7,592.83</b>']
            ]
            
            table = Table(data, colWidths=[3*inch, 1*inch, 1.5*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -4), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (-2, -3), (-1, -1), 'RIGHT'),
                ('FONTNAME', (-2, -3), (-1, -1), 'Helvetica-Bold'),
                ('TEXTCOLOR', (-1, -1), (-1, -1), colors.darkred),
                ('FONTSIZE', (-1, -1), (-1, -1), 14),
            ]))
            story.append(table)
            story.append(Spacer(1, 0.3*inch))
            
            # Signature section
            signature = Paragraph(
                "<b>Authorized Signature:</b><br/>"
                "___________________________<br/>"
                "John Smith, Director of AI<br/>"
                "AI Innovations Inc.",
                styles['Normal']
            )
            story.append(signature)
            story.append(Spacer(1, 0.2*inch))
            
            # Terms and conditions
            terms = Paragraph(
                "<b>Terms & Conditions:</b><br/>"
                "1. Payment due within 30 days<br/>"
                "2. All prices in USD<br/>"
                "3. Support includes 24/7 access<br/>"
                "4. Training scheduled upon payment",
                styles['Normal']
            )
            story.append(terms)
            
            # Add more text content for OCR
            additional = Paragraph(
                "<b>Additional Information:</b><br/>"
                "This document contains invoice data for testing purposes. "
                "The total amount is $7,592.83. The customer is Test Customer Inc. "
                "Invoice number is INV-TEST-001. This should be detected by the OCR system.",
                styles['Normal']
            )
            story.append(additional)
            
            doc.build(story)
            print(f"✅ Created comprehensive test document: {TEST_PDF}")
            print(f"   Size: {Path(TEST_PDF).stat().st_size / 1024:.1f} KB")
            return True
            
        except ImportError:
            print("❌ ReportLab not available. Install with: pip install reportlab")
            return False
        except Exception as e:
            print(f"❌ Error creating document: {e}")
            return False
    
    def check_server_ready(self):
        """Check if server is ready before starting tests."""
        print("\n🔍 Checking server readiness...")
        max_retries = 5
        for i in range(max_retries):
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=10)
                if response.status_code == 200:
                    print("✅ Server is ready")
                    return True
                else:
                    print(f"⚠️  Server returned status {response.status_code}")
            except requests.exceptions.ConnectionError:
                print(f"⚠️  Waiting for server... (attempt {i+1}/{max_retries})")
                time.sleep(3)
            except Exception as e:
                print(f"⚠️  Error checking server: {e}")
                time.sleep(3)
        
        print("❌ Server not ready after multiple attempts")
        return False
    
    def test_phase1(self):
        """Test Phase 1: Document Upload - REAL."""
        print("\n" + "="*70)
        print("📤 PHASE 1: DOCUMENT UPLOAD & INGESTION")
        print("="*70)
        
        if not Path(TEST_PDF).exists():
            if not self.create_test_document():
                return False
        
        try:
            print(f"Uploading {TEST_PDF}...")
            start_time = time.time()
            
            with open(TEST_PDF, 'rb') as f:
                files = {'file': (TEST_PDF, f, 'application/pdf')}
                response = requests.post(
                    f"{BASE_URL}/ingest/upload",
                    files=files,
                    timeout=60  # Increased timeout
                )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                self.document_id = result['document_id']
                
                print(f"✅ PHASE 1 SUCCESSFUL! ({elapsed:.1f}s)")
                print(f"   📋 Document ID: {self.document_id}")
                print(f"   📄 Pages: {result['total_pages']}")
                print(f"   💾 Metadata: {result['metadata_path']}")
                print(f"   📝 Message: {result['message']}")
                
                # Verify files were created
                doc_dir = Path(f"data/documents/{self.document_id}")
                if doc_dir.exists():
                    files_created = list(doc_dir.rglob("*"))
                    print(f"   📁 Files created: {len(files_created)}")
                    for f in files_created[:3]:  # Show first 3 files
                        print(f"      - {f.relative_to(doc_dir)}")
                
                self.results["phase1"] = {
                    "status": "success",
                    "document_id": self.document_id,
                    "pages": result['total_pages'],
                    "time_seconds": elapsed,
                    "files_created": len(files_created) if 'files_created' in locals() else 0
                }
                return True
            else:
                print(f"❌ PHASE 1 FAILED: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            print(f"❌ PHASE 1 ERROR: {e}")
            return False
    
    def test_phase2(self):
        """Test Phase 2: Layout Analysis - REAL YOLOv8."""
        print("\n" + "="*70)
        print("🔍 PHASE 2: LAYOUT ANALYSIS (YOLOv8)")
        print("="*70)
        
        if not self.document_id:
            print("❌ No document ID. Run Phase 1 first.")
            return False
        
        try:
            print(f"Starting layout analysis for {self.document_id}...")
            
            # First check layout model status
            print("   🔍 Checking layout model...")
            model_response = requests.get(
                f"{BASE_URL}/layout/model/info",
                timeout=10
            )
            
            if model_response.status_code == 200:
                model_info = model_response.json()
                print(f"   ✅ Model: {model_info.get('model_name', 'Unknown')}")
                print(f"   📍 Path: {model_info.get('model_path', 'Unknown')}")
                print(f"   🚀 Status: {model_info.get('status', 'Unknown')}")
            else:
                print(f"   ⚠️  Could not get model info: {model_response.status_code}")
            
            start_time = time.time()
            
            # Start layout analysis
            response = requests.post(
                f"{BASE_URL}/layout/analyze/{self.document_id}",
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Layout analysis started")
                print(f"   📋 Status: {result['status']}")
                print(f"   💬 Message: {result['message']}")
                
                # Wait for completion with longer timeout
                print("   ⏳ Waiting for completion (this may take time for YOLOv8)...")
                max_checks = 30  # 30 checks * 3 seconds = 90 seconds max
                
                for i in range(max_checks):
                    time.sleep(3)
                    
                    try:
                        status_response = requests.get(
                            f"{BASE_URL}/layout/status/{self.document_id}",
                            timeout=10
                        )
                        
                        if status_response.status_code == 200:
                            status = status_response.json()
                            current_status = status['status']
                            
                            if current_status == 'completed':
                                elapsed = time.time() - start_time
                                
                                # Get results
                                results_response = requests.get(
                                    f"{BASE_URL}/layout/results/{self.document_id}",
                                    timeout=10
                                )
                                
                                if results_response.status_code == 200:
                                    results = results_response.json()
                                    
                                    # Extract data
                                    if 'pages' in results:
                                        pages = results['pages']
                                        total_detections = sum(len(page.get('detections', [])) for page in pages)
                                        
                                        # Count detection types
                                        detection_types = {}
                                        for page in pages:
                                            for det in page.get('detections', []):
                                                det_type = det.get('type', 'unknown')
                                                detection_types[det_type] = detection_types.get(det_type, 0) + 1
                                        
                                        print(f"✅ PHASE 2 COMPLETED! ({elapsed:.1f}s)")
                                        print(f"   📊 Pages analyzed: {len(pages)}")
                                        print(f"   🎯 Total detections: {total_detections}")
                                        print(f"   📋 Detection types: {detection_types}")
                                        
                                        # Show sample detections
                                        if pages and pages[0].get('detections'):
                                            print(f"   🔍 Sample detections (first page):")
                                            for det in pages[0]['detections'][:5]:
                                                print(f"      - {det.get('type')}: {det.get('bbox')}")
                                        
                                        self.results["phase2"] = {
                                            "status": "success",
                                            "time_seconds": elapsed,
                                            "pages_analyzed": len(pages),
                                            "total_detections": total_detections,
                                            "detection_types": detection_types
                                        }
                                        return True
                                    else:
                                        print(f"   ⚠️  No pages data in results")
                                else:
                                    print(f"   ⚠️  Failed to get results: {results_response.status_code}")
                                
                                break
                            elif current_status == 'failed':
                                error_msg = status.get('error', 'Unknown error')
                                print(f"❌ Layout analysis failed: {error_msg}")
                                break
                            
                            # Show progress
                            if i % 5 == 0:  # Every 15 seconds
                                print(f"   ⏱️  Still processing... ({i*3}s elapsed)")
                        
                    except requests.exceptions.Timeout:
                        print(f"   ⏱️  Check {i+1}: Timeout checking status")
                        continue
                    except Exception as e:
                        print(f"   ⚠️  Check {i+1} error: {e}")
                        continue
                
                print("⚠️  Layout analysis timed out (90+ seconds)")
                return False
            else:
                print(f"❌ Failed to start layout analysis: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            print(f"❌ PHASE 2 ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_phase3(self):
        """Test Phase 3: OCR Processing - REAL EasyOCR."""
        print("\n" + "="*70)
        print("🔤 PHASE 3: OCR PROCESSING (EasyOCR)")
        print("="*70)
        
        if not self.document_id:
            print("❌ No document ID. Run Phase 1 first.")
            return False
        
        # Check if layout analysis was successful first
        if self.results["phase2"]["status"] != "success":
            print("⚠️  Layout analysis not successful. OCR may have no regions to process.")
        
        try:
            print(f"Starting OCR processing for {self.document_id}...")
            
            # Check OCR engine status
            print("   🔍 Checking OCR engine...")
            engine_response = requests.get(
                f"{BASE_URL}/ocr/engine/info",
                timeout=10
            )
            
            if engine_response.status_code == 200:
                engine_info = engine_response.json()
                using_real = engine_info.get('using_real_ocr', False)
                if using_real:
                    print(f"   ✅ Using REAL EasyOCR")
                    print(f"   📊 Languages: {engine_info.get('languages', ['en'])}")
                    print(f"   🚀 GPU: {engine_info.get('gpu_available', False)}")
                else:
                    print(f"   ❌ NOT using real OCR!")
                    return False
            else:
                print(f"   ⚠️  Could not get OCR engine info")
            
            start_time = time.time()
            
            # Start OCR processing
            response = requests.post(
                f"{BASE_URL}/ocr/process/{self.document_id}",
                params={"lang": "en", "use_gpu": False},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ OCR processing started")
                print(f"   🚀 Engine: {result['engine']}")
                print(f"   🎯 Mode: {result['mode']}")
                print(f"   💬 Message: {result['message']}")
                
                # Verify REAL OCR mode
                if result.get('mode') != 'real':
                    print("❌ ERROR: OCR is not in REAL mode!")
                    return False
                
                # Wait for completion
                print("   ⏳ Waiting for OCR processing...")
                max_checks = 40  # 40 checks * 3 seconds = 120 seconds max
                
                for i in range(max_checks):
                    time.sleep(3)
                    
                    try:
                        status_response = requests.get(
                            f"{BASE_URL}/ocr/status/{self.document_id}",
                            timeout=10
                        )
                        
                        if status_response.status_code == 200:
                            status = status_response.json()
                            current_status = status['status']
                            
                            if current_status == 'completed':
                                elapsed = time.time() - start_time
                                
                                # Get results
                                results_response = requests.get(
                                    f"{BASE_URL}/ocr/results/{self.document_id}",
                                    timeout=10
                                )
                                
                                if results_response.status_code == 200:
                                    results = results_response.json()
                                    
                                    # Extract OCR data
                                    pages = results.get('pages', [])
                                    total_regions = sum(page.get('total_regions', 0) for page in pages)
                                    ocr_regions = sum(page.get('ocr_regions', 0) for page in pages)
                                    skipped_regions = sum(page.get('skipped_regions', 0) for page in pages)
                                    
                                    # Get sample text
                                    sample_texts = []
                                    if pages:
                                        for region in pages[0].get('regions', [])[:5]:
                                            text = region.get('ocr_text', '').strip()
                                            if text and len(text) > 10:
                                                sample_texts.append(text[:100])
                                    
                                    print(f"✅ PHASE 3 COMPLETED! ({elapsed:.1f}s)")
                                    print(f"   📊 Pages processed: {len(pages)}")
                                    print(f"   🎯 Total regions: {total_regions}")
                                    print(f"   🔤 OCR regions with text: {ocr_regions}")
                                    print(f"   ⏭️  Skipped regions: {skipped_regions}")
                                    
                                    if sample_texts:
                                        print(f"   📝 Sample text found:")
                                        for i, text in enumerate(sample_texts[:3]):
                                            print(f"      {i+1}. {text}...")
                                    else:
                                        print(f"   ⚠️  No text found in OCR results")
                                        # Debug: show what's in the results
                                        if pages and pages[0].get('regions'):
                                            print(f"   🔍 Regions found but no text:")
                                            for i, region in enumerate(pages[0]['regions'][:3]):
                                                print(f"      Region {i}: {region.get('bbox')}")
                                    
                                    # Verify REAL OCR again
                                    engine_response = requests.get(
                                        f"{BASE_URL}/ocr/engine/info",
                                        timeout=5
                                    )
                                    if engine_response.status_code == 200:
                                        engine_info = engine_response.json()
                                        if engine_info.get('using_real_ocr'):
                                            print(f"   ✅ VERIFIED: Using REAL EasyOCR")
                                        else:
                                            print(f"   ❌ ERROR: Not using real OCR")
                                            return False
                                    
                                    self.results["phase3"] = {
                                        "status": "success",
                                        "time_seconds": elapsed,
                                        "pages_processed": len(pages),
                                        "total_regions": total_regions,
                                        "ocr_regions": ocr_regions,
                                        "real_ocr": True
                                    }
                                    return True
                                
                                break
                            elif current_status == 'failed':
                                error_msg = status.get('error', 'Unknown error')
                                print(f"❌ OCR processing failed: {error_msg}")
                                break
                            
                            pages_done = status.get('pages_processed', 0)
                            if i % 5 == 0:  # Every 15 seconds
                                print(f"   ⏱️  Check {i+1}: {current_status} ({pages_done} pages)")
                        
                    except requests.exceptions.Timeout:
                        print(f"   ⏱️  Check {i+1}: Timeout checking status")
                        continue
                    except Exception as e:
                        print(f"   ⚠️  Check {i+1} error: {e}")
                        continue
                
                print("⚠️  OCR processing timed out (120+ seconds)")
                return False
            else:
                print(f"❌ Failed to start OCR: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            print(f"❌ PHASE 3 ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_phase4(self):
        """Test Phase 4: Multi-Modal Agents - REAL LangGraph."""
        print("\n" + "="*70)
        print("🧠 PHASE 4: MULTI-MODAL AGENTS (LangGraph)")
        print("="*70)
        
        if not self.document_id:
            print("❌ No document ID. Run Phase 1 first.")
            return False
        
        # Check prerequisites
        if self.results["phase2"]["status"] != "success":
            print("⚠️  Layout analysis not successful. Agents may not work properly.")
        if self.results["phase3"]["status"] != "success":
            print("⚠️  OCR not successful. Agents will have no text data.")
        
        try:
            print(f"Starting agent pipeline for {self.document_id}...")
            print("This will run 4 LLM-powered agents with LangGraph orchestration")
            print("Agents: Vision → Text → Fusion → Validation")
            
            # Check LLM configuration
            print("   🔍 Checking LLM configuration...")
            llm_response = requests.get(
                f"{BASE_URL}/agents/llm/info",
                timeout=10
            )
            
            if llm_response.status_code == 200:
                llm_info = llm_response.json()
                print(f"   ✅ LLM: {llm_info.get('model', 'Unknown')}")
                print(f"   📍 Provider: {llm_info.get('llm_provider', 'Unknown')}")
                print(f"   🔧 Config: {llm_info.get('config_status', 'Unknown')}")
            else:
                print(f"   ⚠️  Could not get LLM info: {llm_response.status_code}")
            
            start_time = time.time()
            
            # Start agent pipeline
            response = requests.post(
                f"{BASE_URL}/agents/run/{self.document_id}",
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Agent pipeline started")
                print(f"   🤖 Agents: {', '.join(result['agents'])}")
                print(f"   🔄 Workflow: {result['workflow']}")
                print(f"   🧠 LLM: {result['llm']}")
                print(f"   💾 Vector DB: {result['vector_db']}")
                print(f"   💬 Message: {result['message']}")
                
                # Wait for completion
                print("\n   ⏳ Waiting for agent processing...")
                print("   (This is the INTELLIGENCE CORE - LLMs processing document)")
                
                max_checks = 80  # 80 checks * 5 seconds = 400 seconds max
                last_progress = ""
                
                for i in range(max_checks):
                    time.sleep(5)
                    
                    try:
                        status_response = requests.get(
                            f"{BASE_URL}/agents/status/{self.document_id}",
                            timeout=15
                        )
                        
                        if status_response.status_code == 200:
                            status = status_response.json()
                            current_status = status['status']
                            agents_done = len(status.get('agents_executed', []))
                            
                            progress = f"Status: {current_status}, Agents: {agents_done}/4"
                            if progress != last_progress:
                                print(f"   ⏱️  Check {i+1}: {progress}")
                                last_progress = progress
                            
                            if current_status == 'completed':
                                elapsed = time.time() - start_time
                                
                                # Get results
                                results_response = requests.get(
                                    f"{BASE_URL}/agents/result/{self.document_id}",
                                    timeout=15
                                )
                                
                                if results_response.status_code == 200:
                                    results = results_response.json()
                                    final_output = results.get('final_output', {})
                                    
                                    print(f"\n✅ PHASE 4 COMPLETED! ({elapsed:.1f}s)")
                                    print(f"   🎯 Confidence: {final_output.get('confidence_score', 0):.2%}")
                                    print(f"   📋 Document type: {final_output.get('document_type', 'unknown')}")
                                    print(f"   🔍 Extracted fields: {len(final_output.get('extracted_fields', {}))}")
                                    
                                    # Show extracted fields
                                    fields = final_output.get('extracted_fields', {})
                                    if fields:
                                        print(f"\n   📊 EXTRACTED DATA:")
                                        for key, value in list(fields.items())[:8]:
                                            if isinstance(value, dict):
                                                val = value.get('value', 'N/A')
                                                conf = value.get('confidence', 0)
                                                source = value.get('source', 'unknown')
                                                print(f"     • {key}: {val[:50]}... (confidence: {conf:.2%}, source: {source})")
                                            else:
                                                print(f"     • {key}: {value[:50]}...")
                                    
                                    # Test RAG query
                                    print(f"\n   🔍 Testing RAG query...")
                                    query_response = requests.post(
                                        f"{BASE_URL}/agents/query",
                                        json={
                                            "query": "What is the total amount in this document?",
                                            "document_id": self.document_id
                                        },
                                        timeout=10
                                    )
                                    
                                    if query_response.status_code == 200:
                                        query_result = query_response.json()
                                        results_count = len(query_result.get('results', []))
                                        print(f"   ✅ RAG query returned {results_count} results")
                                        if results_count > 0:
                                            print(f"   📝 Answer: {query_result.get('results', [{}])[0].get('content', 'No content')[:100]}...")
                                    
                                    # Check vector DB
                                    vector_response = requests.get(
                                        f"{BASE_URL}/agents/vector/info",
                                        timeout=5
                                    )
                                    if vector_response.status_code == 200:
                                        vector_info = vector_response.json()
                                        print(f"   💾 Vector DB: {vector_info.get('vector_db', 'Unknown')}")
                                        print(f"   📊 Status: {vector_info.get('status', 'Unknown')}")
                                    
                                    self.results["phase4"] = {
                                        "status": "success",
                                        "time_seconds": elapsed,
                                        "confidence": final_output.get('confidence_score', 0),
                                        "document_type": final_output.get('document_type', 'unknown'),
                                        "extracted_fields": len(fields),
                                        "agents_executed": agents_done
                                    }
                                    return True
                                
                            elif current_status == 'failed':
                                error_msg = status.get('error', 'Unknown error')
                                print(f"❌ Agent pipeline failed: {error_msg}")
                                break
                    
                    except requests.exceptions.Timeout:
                        if i % 10 == 0:  # Every 50 seconds
                            print(f"   ⏱️  Check {i+1}: Timeout (still processing)")
                        continue
                    except Exception as e:
                        print(f"   ⚠️  Check {i+1} error: {e}")
                        continue
                
                print("⚠️  Agent pipeline timed out (400+ seconds)")
                return False
            else:
                print(f"❌ Failed to start agent pipeline: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            print(f"❌ PHASE 4 ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def verify_system_components(self):
        """Verify all system components are working."""
        print("\n" + "="*70)
        print("🔧 SYSTEM COMPONENT VERIFICATION")
        print("="*70)
        
        components = [
            ("Health", "/health", "GET"),
            ("System Info", "/info", "GET"),
            ("OCR Engine", "/ocr/engine/info", "GET"),
            ("LLM Config", "/agents/llm/info", "GET"),
            ("Vector DB", "/agents/vector/info", "GET"),
            ("Model Info", "/layout/model/info", "GET"),
        ]
        
        all_ok = True
        for name, endpoint, method in components:
            try:
                response = requests.request(method, f"{BASE_URL}{endpoint}", timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if name == "Health":
                        print(f"✅ {name}: Server is running")
                    elif name == "OCR Engine":
                        real_ocr = data.get('using_real_ocr', False)
                        engine = data.get('engine', 'Unknown')
                        status = "REAL OCR" if real_ocr else "NOT REAL"
                        print(f"✅ {name}: {engine} - {status}")
                        if not real_ocr:
                            all_ok = False
                    elif name == "LLM Config":
                        model = data.get('model', 'Unknown')
                        provider = data.get('llm_provider', 'Unknown')
                        status = data.get('config_status', 'Unknown')
                        print(f"✅ {name}: {model} ({provider}) - {status}")
                        if status != 'configured':
                            all_ok = False
                    elif name == "Vector DB":
                        db = data.get('vector_db', 'Unknown')
                        status = data.get('status', 'Unknown')
                        print(f"✅ {name}: {db} - {status}")
                    else:
                        print(f"✅ {name}: Working")
                else:
                    print(f"❌ {name}: Failed ({response.status_code})")
                    all_ok = False
                    
            except requests.exceptions.Timeout:
                print(f"⚠️  {name}: Timeout checking")
                all_ok = False
            except Exception as e:
                print(f"❌ {name}: Error ({e})")
                all_ok = False
        
        return all_ok
    
    def generate_final_report(self):
        """Generate final test report."""
        print("\n" + "="*70)
        print("📊 FINAL TEST REPORT")
        print("="*70)
        
        total_tests = 0
        passed_tests = 0
        total_time = 0
        
        print("\nPHASE RESULTS:")
        print("-" * 70)
        
        for phase, result in self.results.items():
            status = result.get('status', 'not_started')
            
            if status == 'success':
                symbol = "✅"
                passed_tests += 1
                phase_time = result.get('time_seconds', 0)
                total_time += phase_time
                
                if phase == "phase1":
                    details = f"Document: {result.get('document_id', 'N/A')}, Pages: {result.get('pages', 0)}"
                elif phase == "phase2":
                    details = f"Detections: {result.get('total_detections', 0)}, Types: {result.get('detection_types', {})}"
                elif phase == "phase3":
                    details = f"OCR Regions: {result.get('ocr_regions', 0)}, Real OCR: {result.get('real_ocr', False)}"
                elif phase == "phase4":
                    details = f"Confidence: {result.get('confidence', 0):.2%}, Fields: {result.get('extracted_fields', 0)}"
                else:
                    details = ""
                
                print(f"{symbol} {phase.upper():10} SUCCESS ({phase_time:.1f}s)")
                print(f"     {details}")
                
            elif status == 'failed':
                symbol = "❌"
                print(f"{symbol} {phase.upper():10} FAILED")
            else:
                symbol = "⚠️"
                print(f"{symbol} {phase.upper():10} NOT TESTED")
            
            total_tests += 1
        
        print("-" * 70)
        print(f"\n📈 SUMMARY:")
        print(f"   Total phases: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Success rate: {(passed_tests/total_tests*100):.1f}%")
        print(f"   Total time: {total_time:.1f} seconds")
        
        if self.document_id:
            print(f"\n📁 GENERATED DATA:")
            doc_dir = Path(f"data/documents/{self.document_id}")
            if doc_dir.exists():
                for phase_dir in sorted(doc_dir.iterdir()):
                    if phase_dir.is_dir():
                        files = list(phase_dir.rglob("*"))
                        if files:
                            size_kb = sum(f.stat().st_size for f in files) / 1024
                            print(f"   📂 {phase_dir.name}: {len(files)} files ({size_kb:.1f} KB)")
        
        print("\n🎯 SYSTEM VERIFICATION:")
        components_ok = self.verify_system_components()
        
        print("\n" + "="*70)
        if passed_tests == 4:
            print("🎉🎉🎉 ALL 4 PHASES SUCCESSFUL! 🎉🎉🎉")
            print("\n✅ Document Intelligence System is FULLY OPERATIONAL")
            print("✅ Real OCR working (EasyOCR)")
            print("✅ Real AI Agents working (LangGraph)")
            print("✅ Real Computer Vision (YOLOv8)")
            print("✅ Real Document Understanding")
            
            print(f"\n📋 Document processed: {self.document_id}")
            print(f"📁 Check results in: data/documents/{self.document_id}/")
            print(f"📚 API Documentation: {BASE_URL}/docs")
            
        elif passed_tests >= 2:
            print("⚠️  PARTIAL SUCCESS - Some phases working")
            print(f"   {passed_tests}/4 phases completed")
            print("\n🔍 Debug suggestions:")
            if self.results["phase2"]["status"] != "success":
                print("   • Check YOLOv8 model installation")
                print("   • Verify CUDA/GPU if using GPU")
            if self.results["phase3"]["status"] != "success":
                print("   • Check EasyOCR installation")
                print("   • Verify layout analysis produced regions")
            if self.results["phase4"]["status"] != "success":
                print("   • Check LLM configuration")
                print("   • Verify vector database is running")
        else:
            print("❌ SYSTEM NOT FULLY OPERATIONAL")
            print("   Check individual phase errors above")
        
        print("=" * 70)
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        print("\n" + "="*70)
        print("🚀 COMPLETE DOCUMENT INTELLIGENCE SYSTEM TEST")
        print("="*70)
        print("Testing ALL 4 phases with REAL execution (NO MOCKING)")
        print("Phase 1: Document Upload & Ingestion")
        print("Phase 2: Layout Analysis (YOLOv8) - REAL")
        print("Phase 3: OCR Processing (EasyOCR) - REAL")
        print("Phase 4: Multi-Modal Agents (LangGraph) - REAL")
        print("="*70)
        
        # Check if server is running
        if not self.check_server_ready():
            print("❌ Server not ready. Start server with: uvicorn app.main:app --host 0.0.0.0 --port 8000")
            return False
        
        # Run tests in sequence
        tests = [
            ("Phase 1", self.test_phase1),
            ("Phase 2", self.test_phase2),
            ("Phase 3", self.test_phase3),
            ("Phase 4", self.test_phase4),
        ]
        
        for test_name, test_func in tests:
            print(f"\n▶️  Starting {test_name}...")
            success = test_func()
            
            if not success:
                print(f"❌ {test_name} failed")
                # Still try to continue to see what else works
        
        # Generate final report
        self.generate_final_report()
        
        return True

def main():
    """Main function."""
    tester = CompleteTester()
    
    try:
        tester.run_all_tests()
        
        print("\n" + "="*70)
        print("🎯 TEST COMPLETED")
        print("="*70)
        
        if tester.document_id:
            print(f"\n📋 Your test document: {tester.document_id}")
            print("\n🔗 Additional test commands:")
            print(f"  curl {BASE_URL}/agents/result/{tester.document_id} | python -m json.tool")
            print(f"  curl {BASE_URL}/ocr/results/{tester.document_id} | python -m json.tool")
            print(f"  curl {BASE_URL}/layout/results/{tester.document_id} | python -m json.tool")
            print(f"\n📚 Full API docs: {BASE_URL}/docs")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()