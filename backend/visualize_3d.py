"""
visualize_3d.py
═══════════════
Extracts embeddings from ChromaDB, reduces to 3D via PCA,
and generates an interactive Plotly 3D scatter in an HTML file.
"""

import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import json
from database import _get_vector_store, USE_PINECONE, PINECONE_INDEX_NAME

# ── 1. Load data from vector store ────────────────────────────────────────────
def generate_3d_viz():
    if USE_PINECONE:
        print(f"🌲 Fetching data from Pinecone Index: {PINECONE_INDEX_NAME}…")
        store = _get_vector_store()
        results = store.query(
            vector=[0.1] + [0.0] * 1535, # non-zero dummy vector for Cosine metric
            top_k=100,
            include_metadata=True,
            include_values=True
        )
        matches = results.matches
        ids = [m.id for m in matches]
        docs = [m.metadata.get("text", "") for m in matches]
        metas = [m.metadata for m in matches]
        embeddings = [m.values for m in matches]
    else:
        print("💾 Connecting to Local ChromaDB…")
        store = _get_vector_store()
        data = store.get(include=["documents", "metadatas", "embeddings"])
        ids       = data["ids"]
        docs      = data["documents"]
        metas     = data["metadatas"]
        embeddings = data["embeddings"]

    if not ids:
        print("⚠️  No documents found in ChromaDB. Skipping visualization.")
        return

    if embeddings is None or (hasattr(embeddings, '__len__') and len(embeddings) == 0):
        print("⚠️  No embeddings found. ChromaDB may not store raw vectors by default.")
        print("   Falling back to generating embeddings via OpenAI…")

        import os
        from dotenv import load_dotenv
        load_dotenv()
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print(f"   Embedding {len(docs)} documents (this may cost a few cents)…")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[d[:500] for d in docs],  # truncate to avoid token limits
        )
        embeddings = [r.embedding for r in response.data]
        print("   ✅ Embeddings generated.")

    embeddings = np.array(embeddings)
    print(f"✅ Loaded {len(ids)} vectors  |  dimensionality: {embeddings.shape[1]}")

    # ── 2. PCA → 3D ───────────────────────────────────────────────────────────────
    print("📐 Reducing to 3D with PCA…")
    n_components = min(3, len(ids))
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(embeddings)

    variance = pca.explained_variance_ratio_ * 100
    print(f"   PCA variance explained: PC1={variance[0]:.1f}%  PC2={variance[1]:.1f}%  PC3={variance[2] if len(variance) > 2 else 0:.1f}%")

    # ── 3. Build labels and colors by source/topic ────────────────────────────────
    source_map = {
        "web_search_credible": "#10b981",  # green
        "manual_upload":       "#a78bfa",  # purple
        "pdf_upload":          "#f43f5e",  # rose
    }

    labels, colors, hover_texts = [], [], []
    for i in range(len(ids)):
        meta = metas[i] or {}
        source = meta.get("source", "unknown")
        question = meta.get("question", "unknown")
        freshness = meta.get("freshness", "")[:19]
        text_preview = (docs[i] or "")[:120].replace("<", "&lt;")

        labels.append(ids[i][:8])
        colors.append(source_map.get(source, "#64748b"))
        hover_texts.append(
            f"<b>ID:</b> {ids[i][:8]}…<br>"
            f"<b>Source:</b> {source}<br>"
            f"<b>Question:</b> {question[:60]}<br>"
            f"<b>Freshness:</b> {freshness}<br>"
            f"<b>Text:</b> {text_preview}…"
        )

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2] if coords.shape[1] > 2 else np.zeros(len(ids))

    # ── 4. Build Plotly figure ────────────────────────────────────────────────────
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(
            size=7,
            color=colors,
            opacity=0.88,
            line=dict(width=0.5, color="#0f172a"),
        ),
        hovertext=hover_texts,
        hoverinfo="text",
        text=labels,
    )])

    fig.update_layout(
        title=dict(
            text=f"⚡ ACARA — Vector Memory 3D ({len(ids)} docs)<br>"
                 f"<sup>PCA variance: {variance[0]:.1f}% + {variance[1]:.1f}% + {variance[2] if len(variance) > 2 else 0:.1f}%"
                 f" = {sum(variance[:3]):.1f}% total</sup>",
            font=dict(size=16, color="#a78bfa"),
        ),
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#0f0f1a",
        scene=dict(
            bgcolor="#0f0f1a",
            xaxis=dict(title="PC1", backgroundcolor="#0f0f1a", color="#94a3b8", gridcolor="#1e293b"),
            yaxis=dict(title="PC2", backgroundcolor="#0f0f1a", color="#94a3b8", gridcolor="#1e293b"),
            zaxis=dict(title="PC3", backgroundcolor="#0f0f1a", color="#94a3b8", gridcolor="#1e293b"),
        ),
        font=dict(color="#e2e8f0"),
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(bgcolor="#1e293b"),
        annotations=[
            dict(
                text="🟢 web_search_credible &nbsp;&nbsp; 🟣 manual_upload &nbsp;&nbsp; 🔴 pdf_upload <br>Hover over dots for details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.01, y=0.01,
                font=dict(size=11, color="#94a3b8"),
                bgcolor="#1e293b",
                borderpad=6,
            )
        ],
    )

    # ── 5. Save HTML ───────────────────────────────────────────────────────────────
    output_path = "chroma_3d_viz.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"\n✅ 3D visualization saved → {output_path}")

if __name__ == "__main__":
    generate_3d_viz()

