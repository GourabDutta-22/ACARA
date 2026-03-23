import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="agentic_rag")

results = collection.query(
    query_texts=["what is the score of Computer networks in SHL exam?"],
    n_results=5
)

print("--- Query Results ---")
for i, d in enumerate(results.get("documents", [[]])[0]):
    print(f"[{i}] Distance: {results.get('distances', [[0]])[0][i]}")
    print(f"[{i}] Meta: {results.get('metadatas', [[]])[0][i]}")
    print(f"[{i}] Text: {d[:200]}...")
    print("-" * 40)
