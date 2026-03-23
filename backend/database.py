import os
import chromadb
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient
from langchain_openai import OpenAIEmbeddings

# MongoDB Setup
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URI)
db = client.agentic_rag
feedback_collection = db.feedback
chat_collection = db.chat_history

# Vector Store Config
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "acara-index")

USE_PINECONE = bool(PINECONE_KEY and PINECONE_KEY != "your_pinecone_api_key_here")

# Lazy-loaded resources (don't connect at import time!)
_pinecone_index = None
_chroma_collection = None
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


def _get_vector_store():
    """Return the vector store, initializing it lazily on first call."""
    global _pinecone_index, _chroma_collection

    if USE_PINECONE:
        if _pinecone_index is None:
            from pinecone import Pinecone, ServerlessSpec
            pc = Pinecone(api_key=PINECONE_KEY)
            
            # Check if index exists, create if missing
            if PINECONE_INDEX_NAME not in pc.list_indexes().names():
                print(f"🌲 Creating Pinecone Index: {PINECONE_INDEX_NAME}...")
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=1536, # OpenAI text-embedding-3-small
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
            
            _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            print(f"🌲 Connected to Pinecone Cloud Index: {PINECONE_INDEX_NAME}")
        return _pinecone_index
    else:
        if _chroma_collection is None:
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            _chroma_collection = chroma_client.get_or_create_collection(name="documents")
            print("💾 Using Local ChromaDB")
        return _chroma_collection


def add_document_to_vector_store(doc_id: str, text: str, metadata: dict = None):
    """Upsert a document into the vector store with a freshness timestamp."""
    base_meta = {
        "freshness": datetime.now(timezone.utc).isoformat(),
        "text": text
    }
    if metadata:
        base_meta.update(metadata)

    store = _get_vector_store()

    if USE_PINECONE:
        emb = embeddings_model.embed_query(text)
        store.upsert(vectors=[{"id": doc_id, "values": emb, "metadata": base_meta}])
    else:
        store.upsert(documents=[text], ids=[doc_id], metadatas=[base_meta])


def search_vector_store(query_text: str, n_results: int = 3, embedding: list = None):
    """Query the selected vector store."""
    store = _get_vector_store()

    if USE_PINECONE:
        query_emb = embedding or embeddings_model.embed_query(query_text)
        results = store.query(vector=query_emb, top_k=n_results, include_metadata=True)
        docs  = [m.metadata.get("text", "") for m in results.matches]
        metas = [m.metadata for m in results.matches]
        dists = [1 - m.score for m in results.matches]
        return {"documents": [docs], "metadatas": [metas], "distances": [dists]}
    else:
        count = store.count()
        safe_n = max(1, min(n_results, count)) if count > 0 else 1
        return store.query(
            query_texts=[query_text],
            n_results=safe_n,
            include=["documents", "distances", "metadatas"],
        )


def get_collection_stats() -> dict:
    """Return basic stats about the vector store."""
    try:
        store = _get_vector_store()
        if USE_PINECONE:
            stats = store.describe_index_stats()
            return {"document_count": stats.total_vector_count, "collection_name": PINECONE_INDEX_NAME}
        else:
            count = store.count()
            return {"document_count": count, "collection_name": store.name}
    except Exception:
        return {"document_count": 0, "collection_name": "unknown"}
