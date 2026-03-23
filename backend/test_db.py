from database import add_document_to_chroma, search_chroma
import uuid

try:
    doc_id = str(uuid.uuid4())
    print("Testing add with metadata...")
    add_document_to_chroma(doc_id, "test doc", metadata={"test": "val"})
    print("Testing add without metadata...")
    add_document_to_chroma(str(uuid.uuid4()), "test doc 2", metadata=None)
    
    print("Testing search...")
    res = search_chroma("test")
    print("Search success:", res)
except Exception as e:
    print(f"ERROR: {e}")
