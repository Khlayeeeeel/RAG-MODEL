"""
COMPREHENSIVE MEDICAL RAG - VITALS.CSV EDITION
==============================================
Handles all 11 columns: Date, ID, Nom, Etape, SpO2, Note SpO2, 
Temperature, Note Temp, Tension, Note TA, Flags
"""

import pandas as pd
import chromadb
import google.generativeai as genai
import os
import re
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# ============ SETUP ============
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")

if not GEMINI_API_KEY:
    print("❌ No API key found!")
    exit(1)

print("✅ API key loaded")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="patient_vitals")

# ============ ENHANCED DATA LOADING ============
def load_data():
    """Load vitals.csv with ALL 11 columns properly parsed."""
    print("📊 Loading vitals.csv with full analysis...")
    
    # Load CSV
    if os.path.exists("vitals.csv"):
        df = pd.read_csv("vitals.csv")
    elif os.path.exists("data/vitals.csv"):
        df = pd.read_csv("data/vitals.csv")
    else:
        raise FileNotFoundError("vitals.csv not found!")
    
    print(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Parse special formats
    df = parse_special_formats(df)
    
    # Clear old data
    if collection.count() > 0:
        all_ids = collection.get()["ids"]
        collection.delete(ids=all_ids)
    
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in df.iterrows():
        # Create RICH document with ALL information
        doc_text = create_rich_document(row)
        documents.append(doc_text)
        
        # Create COMPREHENSIVE metadata
        metadata = create_comprehensive_metadata(row)
        metadatas.append(metadata)
        ids.append(f"record_{idx}")
    
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"✅ Added {len(documents)} rich records")
    return df

def parse_special_formats(df):
    """Parse BP (12.2/7.8), Notes (6/10), and Dates."""
    # Parse Tension into systolic/diastolic
    df[['bp_systolic', 'bp_diastolic']] = df['Tension'].str.split('/', expand=True).astype(float)
    
    # Parse Notes into numeric scores
    df['spo2_score'] = df['Note SpO2'].str.split('/').str[0].astype(float)
    df['temp_score'] = df['Note Temp'].str.split('/').str[0].astype(float)
    df['bp_score'] = df['Note TA'].str.split('/').str[0].astype(float)
    
    # Parse Date
    df['datetime'] = pd.to_datetime(df['Date'])
    df['date_only'] = df['datetime'].dt.date.astype(str)
    df['time_only'] = df['datetime'].dt.time.astype(str)
    
    return df

def create_rich_document(row):
    """Create comprehensive text with ALL patient information."""
    return (
        f"PATIENT RECORD\n"
        f"===============\n"
        f"Date/Time: {row['Date']}\n"
        f"Patient: {row['Nom']} (ID: {row['ID']})\n"
        f"Stage: {row['Etape']}\n"
        f"\n"
        f"VITAL SIGNS:\n"
        f"  • SpO2: {row['SpO2']}% [Quality: {row['Note SpO2']}]\n"
        f"  • Temperature: {row['Temperature']}°C [Quality: {row['Note Temp']}]\n"
        f"  • Blood Pressure: {row['Tension']} mmHg [Sys:{row['bp_systolic']}, Dia:{row['bp_diastolic']}] [Quality: {row['Note TA']}]\n"
        f"\n"
        f"ASSESSMENT:\n"
        f"  • Stage: {row['Etape']}\n"
        f"  • Alert Flags: {row['Flags']}/7\n"
        f"  • Data Quality: SpO2({row['Note SpO2']}), Temp({row['Note Temp']}), BP({row['Note TA']})"
    )

def create_comprehensive_metadata(row):
    """Create metadata with ALL fields."""
    return {
        # Identifiers
        "patient_id": str(row['ID']),
        "patient_name": str(row['Nom']),
        
        # Date/Time
        "datetime": str(row['Date']),
        "date": str(row['date_only']),
        "time": str(row['time_only']),
        
        # Stage
        "stage": str(row['Etape']),
        "stage_urgency": 3 if row['Etape'] == 'Urgent' else 2 if row['Etape'] == 'Grave' else 1,
        
        # Vitals
        "spo2_value": float(row['SpO2']),
        "temp_value": float(row['Temperature']),
        "bp_systolic": float(row['bp_systolic']),
        "bp_diastolic": float(row['bp_diastolic']),
        "bp_combined": str(row['Tension']),
        
        # Quality Scores (numeric)
        "spo2_score": float(row['spo2_score']),
        "temp_score": float(row['temp_score']),
        "bp_score": float(row['bp_score']),
        
        # Quality Notes (original text)
        "spo2_note": str(row['Note SpO2']),
        "temp_note": str(row['Note Temp']),
        "bp_note": str(row['Note TA']),
        
        # Flags
        "flags": int(row['Flags']),
        "has_alerts": int(row['Flags']) > 0
    }

# ============ QUESTION HANDLER ============
def ask(question: str) -> str:
    """Handle all question types."""
    print(f"\n📝 Processing: '{question}'")
    q_lower = question.lower()
    
    # Type 1: PATIENT SUMMARY
    if any(x in q_lower for x in ['tell me about', 'summary', 'info on']):
        patient = extract_patient(q_lower)
        if patient:
            return get_patient_summary(patient)
        return "❓ Please specify patient (e.g., 'tell me about Patient_1_73a4')"
    
    # Type 2: SPECIFIC VITAL
    vital_keywords = ['spo2', 'oxygen', 'temperature', 'temp', 'bp', 'blood pressure', 'tension']
    if any(v in q_lower for v in vital_keywords):
        patient = extract_patient(q_lower)
        vital = extract_vital(q_lower)
        if patient:
            return get_specific_vital(patient, vital, 'quality' in q_lower)
        return "❓ Please specify patient"
    
    # Type 3: FILTER
    if any(x in q_lower for x in ['show me', 'find', 'patients with']):
        return filter_patients(q_lower)
    
    # Type 4: ALERTS
    if any(x in q_lower for x in ['abnormal', 'critical', 'alert', 'urgent']):
        return get_alerts()
    
    # Type 5: LIST ALL
    if any(x in q_lower for x in ['all patients', 'list all']):
        return list_all_patients()
    
    # Type 6: GENERAL
    return general_query(question)

def extract_patient(q_lower):
    """Extract patient ID from question."""
    patterns = [r'patient[_-]?([a-zA-Z0-9_]+)', r'id[_-]?([a-zA-Z0-9]+)']
    for p in patterns:
        match = re.search(p, q_lower)
        if match:
            return match.group(1)
    return None

def extract_vital(q_lower):
    """Extract vital type."""
    if any(x in q_lower for x in ['temperature', 'temp']):
        return 'temperature'
    elif any(x in q_lower for x in ['spo2', 'oxygen']):
        return 'spo2'
    elif any(x in q_lower for x in ['bp', 'blood pressure', 'tension']):
        return 'blood_pressure'
    return 'all'

# ============ RETRIEVAL FUNCTIONS ============
def get_patient_summary(patient_ref):
    """Get complete summary for one patient."""
    print(f"🔍 Searching for: {patient_ref}")
    
    # Try exact match
    results = collection.get(where={"patient_id": patient_ref})
    
    if not results or len(results['ids']) == 0:
        # Fuzzy search
        all_data = collection.get()
        docs, metas = [], []
        for doc, meta in zip(all_data['documents'], all_data['metadatas']):
            if patient_ref.lower() in meta['patient_id'].lower() or patient_ref.lower() in meta['patient_name'].lower():
                docs.append(doc)
                metas.append(meta)
    else:
        docs = results['documents']
        metas = results['metadatas']
    
    if not docs:
        return "❌ Patient not found."
    
    print(f"✅ Found {len(docs)} records")
    
    # Build context
    context = "\n\n".join(docs)
    patient_name = metas[0].get('patient_name', 'Unknown')
    
    prompt = f"""You are a medical assistant. Provide a COMPREHENSIVE patient report.

PATIENT: {patient_name}
RECORDS: {len(docs)} visits

ALL INFORMATION:
{context}

Create a detailed medical summary including:
1. Patient identification (Name, ID)
2. Visit timeline (dates and times)
3. Stage classification (Grave/Urgent/Normal)
4. Complete vital signs with quality scores:
   - SpO2 with quality ratings
   - Temperature with quality ratings
   - Blood Pressure with quality ratings
5. Alert flags analysis
6. Trends over time
7. Critical findings

Format as a professional medical report."""
    
    return model.generate_content(prompt).text

def get_specific_vital(patient_ref, vital_type, include_quality):
    """Get specific vital sign for patient."""
    docs, metas = [], []
    
    # Get patient records
    all_data = collection.get()
    for doc, meta in zip(all_data['documents'], all_data['metadatas']):
        if patient_ref.lower() in meta['patient_id'].lower():
            docs.append(doc)
            metas.append(meta)
    
    if not docs:
        return "❌ No records found."
    
    # Build focused context
    vital_name = {'temperature': 'Temperature', 'spo2': 'SpO2', 'blood_pressure': 'Blood Pressure'}.get(vital_type, 'Vitals')
    
    lines = []
    for meta in metas:
        date = meta.get('date', 'Unknown')
        if vital_type == 'temperature':
            val = f"{meta.get('temp_value')}°C"
            quality = meta.get('temp_note') if include_quality else ''
        elif vital_type == 'spo2':
            val = f"{meta.get('spo2_value')}%"
            quality = meta.get('spo2_note') if include_quality else ''
        else:
            val = meta.get('bp_combined')
            quality = meta.get('bp_note') if include_quality else ''
        
        line = f"Date: {date} | {vital_name}: {val}"
        if include_quality:
            line += f" | Quality: {quality}"
        lines.append(line)
    
    context = "\n".join(lines)
    
    prompt = f"""Report on {vital_name} for patient:

{context}

Provide all readings with dates. {'Include quality assessments.' if include_quality else ''} Note any abnormal values."""
    
    return model.generate_content(prompt).text

def filter_patients(q_lower):
    """Filter patients by conditions."""
    # Extract condition (e.g., "SpO2 < 90")
    pattern = r'(spo2|temp|temperature|bp|stage|flags)\s*(<|>|<=|>=|=)\s*(\d+\.?\d*)'
    match = re.search(pattern, q_lower)
    
    if not match:
        # Check for stage name
        for stage in ['urgent', 'grave', 'normal']:
            if stage in q_lower:
                return filter_by_stage(stage.capitalize())
        return "❓ Could not understand filter condition. Try: 'SpO2 < 90' or 'stage = Urgent'"
    
    vital, op, val = match.group(1), match.group(2), float(match.group(3))
    
    all_data = collection.get()
    matching = []
    
    for doc, meta in zip(all_data['documents'], all_data['metadatas']):
        check_val = None
        if vital in ['spo2']: check_val = meta.get('spo2_value')
        elif vital in ['temp', 'temperature']: check_val = meta.get('temp_value')
        elif vital == 'flags': check_val = meta.get('flags')
        
        if check_val is not None:
            include = False
            if op == '<' and check_val < val: include = True
            elif op == '>' and check_val > val: include = True
            elif op == '=' and check_val == val: include = True
            
            if include:
                matching.append(f"{meta.get('patient_name')} (ID: {meta.get('patient_id')}) on {meta.get('date')}: {vital}={check_val}, Stage={meta.get('stage')}, Flags={meta.get('flags')}")
    
    if not matching:
        return f"❌ No patients found with {vital} {op} {val}"
    
    context = "\n".join(matching[:15])  # Limit to 15
    
    prompt = f"""Found {len(matching)} matching records:

{context}

Summarize who these patients are and their condition. Highlight any urgent cases."""
    
    return model.generate_content(prompt).text

def filter_by_stage(stage_name):
    """Filter by stage (Grave/Urgent/Normal)."""
    all_data = collection.get()
    matching = []
    
    for doc, meta in zip(all_data['documents'], all_data['metadatas']):
        if meta.get('stage') == stage_name:
            matching.append(f"{meta.get('patient_name')} (ID: {meta.get('patient_id')}) on {meta.get('date')}: SpO2={meta.get('spo2_value')}%, Temp={meta.get('temp_value')}°C, BP={meta.get('bp_combined')}, Flags={meta.get('flags')}")
    
    if not matching:
        return f"❌ No patients in {stage_name} stage"
    
    return f"📋 PATIENTS IN {stage_name.upper()} STAGE ({len(matching)} records):\n\n" + "\n".join(matching[:10])

def get_alerts():
    """Find critical patients."""
    all_data = collection.get()
    critical = []
    
    for doc, meta in zip(all_data['documents'], all_data['metadatas']):
        reasons = []
        
        # Critical thresholds
        if meta.get('spo2_value', 100) < 90:
            reasons.append(f"Low SpO2: {meta.get('spo2_value')}%")
        if meta.get('temp_value', 37) > 38.5:
            reasons.append(f"Fever: {meta.get('temp_value')}°C")
        if meta.get('bp_systolic', 120) > 160:
            reasons.append(f"High BP: {meta.get('bp_systolic')}")
        if meta.get('flags', 0) >= 3:
            reasons.append(f"High flags: {meta.get('flags')}/7")
        if meta.get('stage') == 'Urgent':
            reasons.append("Stage: Urgent")
        
        if reasons:
            critical.append(f"🚨 {meta.get('patient_name')} ({meta.get('date')}): " + ", ".join(reasons))
    
    if not critical:
        return "✅ No critical alerts. All patients stable."
    
    return "🚨 CRITICAL PATIENTS:\n\n" + "\n".join(critical[:20])

def list_all_patients():
    """List all unique patients."""
    all_data = collection.get()
    
    patients = {}
    for meta in all_data['metadatas']:
        pid = meta.get('patient_id')
        if pid not in patients:
            patients[pid] = {
                'name': meta.get('patient_name'),
                'visits': 0,
                'stages': set()
            }
        patients[pid]['visits'] += 1
        patients[pid]['stages'].add(meta.get('stage'))
    
    lines = [f"📋 ALL PATIENTS ({len(patients)} total):\n"]
    for pid, info in sorted(patients.items()):
        stages = ", ".join(info['stages'])
        lines.append(f"• {info['name']}: {info['visits']} visits [{stages}]")
    
    lines.append(f"\nTotal records: {len(all_data['ids'])}")
    return "\n".join(lines)

def general_query(question):
    """General fallback query."""
    results = collection.query(query_texts=[question], n_results=5)
    
    if not results['documents'][0]:
        return "❌ No relevant records found."
    
    context = "\n\n".join(results['documents'][0])
    
    prompt = f"""Answer this medical question:

Records:
{context}

Question: {question}

Answer based only on these records."""
    
    return model.generate_content(prompt).text

# ============ RUN ============
if __name__ == "__main__":
    load_data()
    
    print("\n" + "="*70)
    print("🩺 COMPLETE MEDICAL RAG - ALL 11 COLUMNS ANALYZED")
    print("="*70)
    print("\n📚 SUPPORTED QUESTIONS:")
    print("\n1️⃣  PATIENT SUMMARY:")
    print("   'tell me about Patient_1_73a4'")
    print("   → Full report with all vitals + quality scores + flags")
    print("\n2️⃣  SPECIFIC VITAL:")
    print("   'What is Patient_1_73a4's SpO2?'")
    print("   'Show temperature with quality for Patient_2_b20f'")
    print("\n3️⃣  FILTER PATIENTS:")
    print("   'Show patients with SpO2 < 90'")
    print("   'Find patients in Urgent stage'")
    print("   'Who has flags > 3?'")
    print("\n4️⃣  CRITICAL ALERTS:")
    print("   'Who has abnormal vitals?'")
    print("   'Show critical patients'")
    print("\n5️⃣  LIST ALL:")
    print("   'Show all patients'")
    print("   'List every patient'")
    print("\n" + "="*70)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ['quit', 'q', 'exit']:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        
        answer = ask(question)
        print(f"\n{'='*70}")
        print(answer)
        print(f"{'='*70}")