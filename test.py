"""
TEST API KEY LOADING
====================
Quick verification that .env file is working correctly
"""

import os
from dotenv import load_dotenv

print("="*60)
print("🔐 API KEY TEST SCRIPT")
print("="*60)

# Step 1: Check if .env file exists
print("\n📁 Checking for .env file...")
if os.path.exists(".env"):
    print("✅ .env file found in current directory")
else:
    print("❌ .env file NOT found!")
    print("   Create a file named '.env' with:")
    print("   GEMINI_API_KEY=your_api_key_here")
    exit(1)

# Step 2: Load .env file
print("\n📂 Loading .env file...")
try:
    load_dotenv()
    print("✅ load_dotenv() executed successfully")
except Exception as e:
    print(f"❌ Error loading .env: {e}")
    exit(1)

# Step 3: Check if key is loaded
print("\n🔑 Checking GEMINI_API_KEY...")
api_key = os.getenv("API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY is EMPTY or NOT SET")
    print("\n   Troubleshooting:")
    print("   1. Check .env file exists in this folder")
    print("   2. Verify format: GEMINI_API_KEY=your_key (no quotes)")
    print("   3. No spaces around = sign")
    exit(1)

# Step 4: Validate key format
print("\n✅ GEMINI_API_KEY is loaded!")
print(f"   Length: {len(api_key)} characters")
print(f"   Starts with: {api_key[:10]}...")
print(f"   Ends with: ...{api_key[-4:]}")

# Check format
if api_key.startswith("AIzaSy"):
    print("\n✅ Key format looks correct (starts with AIzaSy)")
else:
    print("\n⚠️  Warning: Key doesn't start with 'AIzaSy'")
    print("   Gemini keys usually start with 'AIzaSy'")

# Step 5: Optional - Test with Gemini
print("\n" + "="*60)
print("🧪 OPTIONAL: Test API call to Gemini")
print("="*60)

test_api = input("\nTest API key with actual Gemini call? (y/n): ").lower().strip()

if test_api == 'y':
    try:
        import google.generativeai as genai
        
        print("\n🔄 Configuring Gemini...")
        genai.configure(api_key=api_key)
        
        print("🔄 Creating model...")
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        print("🔄 Sending test prompt...")
        response = model.generate_content("Say 'API key is working!' and nothing else.")
        
        print("\n✅ API CALL SUCCESSFUL!")
        print(f"\n🤖 Gemini response: {response.text}")
        
    except Exception as e:
        print(f"\n❌ API CALL FAILED: {e}")
        print("\nPossible issues:")
        print("  • Key is invalid or expired")
        print("  • No internet connection")
        print("  • Gemini API is down")
        print("  • pip install google-generativeai")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)