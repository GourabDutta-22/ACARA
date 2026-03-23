import chromadb
import json
from datetime import datetime, timezone

# ── Connect to ChromaDB ───────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="documents")

data = collection.get(include=["documents", "metadatas"])
ids = data["ids"]
docs = data["documents"]
metas = data["metadatas"]

# ── Terminal output ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"📊 ChromaDB — Collection: 'documents'   Total: {len(ids)} docs")
print(f"{'='*60}\n")

for i in range(len(ids)):
    source = (metas[i] or {}).get("source", "unknown")
    question = (metas[i] or {}).get("question", "—")
    freshness = (metas[i] or {}).get("freshness", "—")
    text_preview = (docs[i] or "")[:200]
    print(f"🆔 {ids[i]}")
    print(f"📝 {text_preview}…")
    print(f"🏷️  source={source} | question={question[:60]}")
    print(f"🕒 {freshness}")
    print("-" * 60)

# ── HTML Report ───────────────────────────────────────────────────────────────
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

rows = ""
for i in range(len(ids)):
    source = (metas[i] or {}).get("source", "unknown")
    question = (metas[i] or {}).get("question", "—")
    freshness = (metas[i] or {}).get("freshness", "—")
    text = (docs[i] or "").replace("<", "&lt;").replace(">", "&gt;")
    badge_color = "#059669" if "credible" in source else "#7c3aed"
    rows += f"""
    <tr>
      <td class="id">{ids[i][:8]}…</td>
      <td class="text">{text[:300]}…</td>
      <td><span class="badge" style="background:{badge_color}">{source}</span></td>
      <td class="meta">{question[:60]}</td>
      <td class="meta">{freshness[:19]}</td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>ACARA — ChromaDB Viewer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f0f1a; color: #e2e8f0; padding: 2rem; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.3rem; color: #a78bfa; }}
  .sub {{ color: #94a3b8; font-size: 0.85rem; margin-bottom: 1.5rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ background: #1e1b4b; color: #c4b5fd; padding: 0.6rem 0.8rem; text-align: left; position: sticky; top: 0; }}
  td {{ padding: 0.55rem 0.8rem; border-bottom: 1px solid #1e293b; vertical-align: top; }}
  tr:hover td {{ background: #1a1a2e; }}
  .id {{ font-family: monospace; color: #7c3aed; white-space: nowrap; }}
  .text {{ max-width: 420px; color: #cbd5e1; }}
  .meta {{ color: #94a3b8; white-space: nowrap; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; color: #fff; }}
  .count {{ display: inline-block; background: #7c3aed; color: #fff; padding: 4px 14px; border-radius: 999px; font-size: 0.9rem; margin-bottom: 1rem; }}
</style>
</head>
<body>
  <h1>⚡ ACARA — ChromaDB Vector Store</h1>
  <div class="sub">Generated: {now}</div>
  <div class="count">{len(ids)} documents</div>
  <table>
    <thead>
      <tr>
        <th>ID</th><th>Text Preview</th><th>Source</th><th>Origin Query</th><th>Freshness</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>"""

with open("chroma_viewer.html", "w") as f:
    f.write(html)

print(f"\n✅ HTML report saved → chroma_viewer.html")
print("   Open it in your browser to see a visual table of all docs.\n")
