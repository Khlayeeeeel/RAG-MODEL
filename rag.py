import google.generativeai as genai

# Test if libraries load
print("✅ Libraries loaded successfully!")

# Test API Connection
try:
    genai.configure(api_key="AIzaSyASnDj74I9Ni4tKbvRdjRnTfC-NHw4R9rI")
    
    # 🔄 UPDATED: Using the current generation model instead of the retired 1.5
    model = genai.GenerativeModel('gemini-2.5-flash') 
    
    response = model.generate_content("Is the RAG system ready?")
    print("✅ Google AI Studio connection: Active")
    print(f"🤖 AI Response: {response.text}")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")