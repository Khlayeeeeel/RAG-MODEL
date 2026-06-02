"""
SIMPLE QUERY SCRIPT
==================
Just run: python query.py "Your question here"
"""

import sys
from rag_simple import ask, collection, load_data

# Auto-load if empty
if collection.count() == 0:
    load_data()

# Get question from command line or prompt
if len(sys.argv) > 1:
    question = " ".join(sys.argv[1:])
else:
    question = input("Ask about patient vitals: ")

print(f"\n🤖 Answer: {ask(question)}")