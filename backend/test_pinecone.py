import os
from dotenv import load_dotenv
load_dotenv()
from database import USE_PINECONE, add_document_to_vector_store, search_vector_store

print("USE_PINECONE:", USE_PINECONE)

if USE_PINECONE:
    print("Testing Pinecone connection and upsert...")
    try:
        add_document_to_vector_store("test_id_123", "This is a test document about artificial intelligence.", {"source": "test"})
        print("Upsert successful.")
        
        print("Testing Pinecone search...")
        results = search_vector_store("What is AI?", n_results=1)
        print("Search results:", results)
        
    except Exception as e:
        print("Error:", repr(e))
else:
    print("Pinecone is not enabled. Please check .env")
