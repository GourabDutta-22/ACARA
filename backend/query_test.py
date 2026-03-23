from database import vector_store

print("Total docs:", vector_store.count())
results = vector_store.query(
    query_texts=["what is the score of Computer networks in SHL exam?"],
    n_results=5,
    include=["documents", "distances", "metadatas"]
)

print("\n--- Top 5 Results for Query ---")
docs = results.get("documents", [[]])[0]
dists = results.get("distances", [[]])[0]
metas = results.get("metadatas", [[]])[0]

for i in range(len(docs)):
    print(f"[{i}] Distance: {dists[i]:.4f}")
    print(f"[{i}] Meta: {metas[i]}")
    print(f"[{i}] Text: {docs[i][:200]}...")
    print("-" * 40)

print("\n--- Search for 'Computer networks' ---")
all_docs = vector_store.get(include=["documents", "metadatas"])
docs = all_docs.get("documents", [])
metas = all_docs.get("metadatas", [])

found = False
for i, d in enumerate(docs):
    if "Computer network" in d or "Computer Network" in d or "computer network" in d:
        print(f"Found in DOC {i} (Meta: {metas[i]})")
        print(d)
        print("="*50)
        found = True

if not found:
    print("Could not find the phrase 'Computer network' in any chunk.")
