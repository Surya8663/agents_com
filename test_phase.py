# deployment_checklist.py
print("📋 DEPLOYMENT READINESS CHECKLIST")
print("=" * 60)

checks = {
    "1. Server running": "http://localhost:8000/health",
    "2. LLM configured": "http://localhost:8000/agents/llm/info", 
    "3. OCR engine ready": "http://localhost:8000/ocr/engine/info",
    "4. Layout model ready": "http://localhost:8000/layout/model/info",
    "5. Vector DB ready": "http://localhost:8000/agents/vector/info",
}
import requests
all_passed = True

for check_name, endpoint in checks.items():
    try:
        response = requests.get(endpoint, timeout=5)
        if response.status_code == 200:
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name} (Status: {response.status_code})")
            all_passed = False
    except Exception as e:
        print(f"❌ {check_name} (Error: {e})")
        all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("🎉 SYSTEM IS DEPLOYMENT READY!")
    print("\n🚀 Next steps:")
    print("   1. Dockerize the application")
    print("   2. Set up environment variables")
    print("   3. Configure production database")
    print("   4. Set up monitoring and logging")
    print("   5. Deploy to cloud (AWS/GCP/Azure)")
else:
    print("⚠️ System needs configuration before deployment")