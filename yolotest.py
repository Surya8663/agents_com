# verify_yolo.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🔍 Verifying YOLOv8 Model")
print("=" * 60)

try:
    from ultralytics import YOLO
    
    print("1. Checking if yolov8n.pt exists...")
    if os.path.exists("yolov8n.pt"):
        print(f"   ✅ Found: yolov8n.pt ({os.path.getsize('yolov8n.pt') / 1024/1024:.1f} MB)")
        
        print("\n2. Trying to load model...")
        try:
            model = YOLO("yolov8n.pt")
            print("   ✅ Model loaded successfully!")
            
            # Test with a dummy image
            print("\n3. Testing with dummy inference...")
            import numpy as np
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            results = model(dummy_image, verbose=False)
            print("   ✅ Inference successful!")
            
            print("\n4. Model info:")
            print(f"   Model type: {type(model).__name__}")
            if hasattr(model, 'names') and model.names:
                print(f"   Number of classes: {len(model.names)}")
                print(f"   Classes: {list(model.names.values())[:5]}...")
            
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   ❌ yolov8n.pt not found")
        
except ImportError as e:
    print(f"❌ ultralytics not installed: {e}")
    print("   Install with: pip install ultralytics")
except Exception as e:
    print(f"❌ Error: {e}")