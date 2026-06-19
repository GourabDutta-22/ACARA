"""
Adaptive Context-Aware Retrieval Agent (LangChain LCEL)
========================================================
Flow Diagram components implemented here:

  User Query → Query Encoder → Vector Memory
                                     ↓
                          Context Awareness Gate
                         (Similarity + Coverage + Freshness)
                         ↙ Context Weak          ↘ Context Strong
         External Knowledge Source           Context Builder
                 ↓                                 ↓
        Dynamic Chunking Module            Generator Model (LLM)
                 ↓                                 ↓
         Embedding Model                  Critic / Validator
                 ↓                                 ↓
    Credibility Scoring → Memory Update   ←   Final Output
                                 ↑
                  Adaptive Retrieval Controller (ARC)

Orchestration: Pure LangChain LCEL (RunnableSequence + RunnableBranch).
LangGraph is NOT used here — it is only used in visualize_3d.py.
"""

import os
import uuid
import asyncio
from datetime import datetime, timezone, timedelta
from typing import TypedDict, Annotated, Sequence, Optional, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database import search_vector_store, add_document_to_vector_store, USE_PINECONE, get_embeddings_model
from arc import arc
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Agent State (plain dict — passed through the LCEL chain)
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    question: str
    documents: list
    original_documents: list          # Original chunks retrieved from DB
    metadatas: list                   # freshness + source per document
    distances: list                   # cosine distances from vector store
    web_fallback: bool
    freshness_ok: bool
    needs_feedback: bool
    final_answer: str
    arc_params: dict                  # snapshot of ARC params for this run
    pipeline_steps: List[str]         # ordered list of steps taken


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class Scorer(BaseModel):
    score: str = Field(description="Binary score 'yes' or 'no'.")
    reasoning: str = Field(description="Brief reasoning for the score.")

structured_llm = llm.with_structured_output(Scorer)
search_tool = TavilySearch(max_results=5)

# ─────────────────────────────────────────────────────────────────────────────
# Session Memory (replaces LangGraph MemorySaver)
# Keyed by session_id → list of BaseMessage
# ─────────────────────────────────────────────────────────────────────────────
_session_memory: dict[str, list[BaseMessage]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# SSE Event Queue (module-level so main.py can subscribe)
# ─────────────────────────────────────────────────────────────────────────────
_event_queues: dict[str, asyncio.Queue] = {}

def get_or_create_queue(session_id: str) -> asyncio.Queue:
    if session_id not in _event_queues:
        _event_queues[session_id] = asyncio.Queue()
    return _event_queues[session_id]

def emit_event(session_id: str, step: str, detail: str = ""):
    """Push a pipeline step event into the session's SSE queue."""
    if session_id in _event_queues:
        try:
            _event_queues[session_id].put_nowait({"step": step, "detail": detail})
        except asyncio.QueueFull:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# LCEL State-Merge Helper
# Each node returns a partial dict; this helper merges it into the full state
# so the next node in the chain receives a complete AgentState dict.
# ─────────────────────────────────────────────────────────────────────────────
def _wrap_node(fn):
    """Wrap a node function so it merges its output back into the incoming state."""
    def wrapped(state: dict) -> dict:
        updates = fn(state)
        return {**state, **updates}
    return RunnableLambda(wrapped)


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — Query Encoder + Vector Memory Retrieval
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_node(state: AgentState):
    """
    User Query → Query Encoder → Vector Memory
    Uses the per-request ARC snapshot (state['arc_params']) for top_k so that
    concurrent sessions do not clobber each other's retrieval parameters.
    """
    question = state["question"]
    # ── Read parameters from the per-request snapshot, NOT the live global ──
    snap       = state.get("arc_params", {})
    session_id = snap.get("_session_id", "")
    top_k      = snap.get("top_k", arc.top_k)  # isolated to this request

    emit_event(session_id, "retrieve", f"Encoding query and searching Vector Memory (top-k={top_k})")

    # Still call the global adjuster so ARC *learns* from query length —
    # but the updated chunk values are re-snapshotted below, not read live.
    arc.adjust_chunk_size(question)

    threshold = snap.get("similarity_threshold", arc.similarity_threshold)

    def do_search(q: str):
        emb = get_embeddings_model().embed_query(q) if USE_PINECONE else None
        res = search_vector_store(q, n_results=top_k, embedding=emb)
        docs = res.get("documents", [[]])[0] or []
        metas = res.get("metadatas", [[]])[0] or []
        dists = res.get("distances", [[]])[0] or []
        return docs, metas, dists

    # 1. Initial fast search
    documents, metadatas, distances = do_search(question)

    # Calculate best score
    best_score = 0.0
    if distances:
        if USE_PINECONE:
            best_score = max(1 - d for d in distances)
        else:
            best_score = max(1 / (1 + d) for d in distances)

    # 2. Lazy Query Expansion (only if initial search is weak)
    if best_score < threshold and len(question.split()) < 10:
        emit_event(session_id, "retrieve", f"Initial search weak (score {best_score:.2f} < {threshold}). Expanding query for better recall.")
        expansion_prompt = PromptTemplate(
            template="You are a medical query expander. Given the short query below, expand it with 3-4 highly relevant medical synonyms or related terms to improve vector search recall. Output ONLY the original query followed by the synonyms on a single line, nothing else. Query: {question}",
            input_variables=["question"]
        )
        try:
            expanded = (expansion_prompt | llm).invoke({"question": question}).content.strip()
            if expanded:
                # 3. Second search with expanded query
                documents, metadatas, distances = do_search(expanded)
        except Exception:
            pass

    return {
        "documents": documents,
        "metadatas": metadatas,
        "distances": distances,
        # Re-snapshot after adjust_chunk_size so downstream nodes (web_search)
        # get the query-tuned chunk size for THIS request only.
        "arc_params": arc.get_params() | {"_session_id": session_id},
        "pipeline_steps": ["retrieve"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — Context Awareness Gate
#           Similarity Check + Coverage Check + Freshness Check
# ─────────────────────────────────────────────────────────────────────────────
FRESHNESS_HOURS = 72  # chunks older than 72 h are considered stale

def grade_documents_node(state: AgentState):
    """
    Context Awareness Gate:
      • Similarity Check  — cosine distance vs ARC threshold
      • Coverage Check    — LLM judges relevance / coverage
      • Freshness Check   — ISO timestamp age > FRESHNESS_HOURS → stale
    """
    question = state["question"]
    documents = state.get("documents", [])
    metadatas = state.get("metadatas", [])
    distances = state.get("distances", [])
    session_id = state.get("arc_params", {}).get("_session_id", "")
    steps = list(state.get("pipeline_steps", []))

    emit_event(session_id, "grade", "Running Context Awareness Gate (Similarity + Coverage + Freshness)")

    if not documents:
        arc.adjust_on_weak_context()
        return {"web_fallback": True, "freshness_ok": False, "pipeline_steps": steps + ["grade"]}

    # ── Similarity Check ──────────────────────────────────────────────────
    # Read from the per-request snapshot — never from the live global —
    # so a concurrent session's adjust_on_weak/strong_context() call cannot
    # change the threshold we use mid-pipeline.
    threshold = state.get("arc_params", {}).get("similarity_threshold", arc.similarity_threshold)

    similarity_pass = False
    best_score = 0.0
    if distances:
        if USE_PINECONE:
            # Pinecone: distance = 1 - cosine_score. We want cosine_score >= threshold
            scores = [(1 - d) for d in distances]
        else:
            # ChromaDB L2 distance: lower = more similar
            scores = [(1 / (1 + d)) for d in distances]
        
        if scores:
            best_score = max(scores)
            similarity_pass = best_score >= threshold

    # ── Freshness Check ───────────────────────────────────────────────────
    freshness_ok = True
    if metadatas:
        now = datetime.now(timezone.utc)
        stale_count = 0
        for meta in metadatas:
            if meta and meta.get("freshness"):
                try:
                    stored_at = datetime.fromisoformat(meta["freshness"])
                    if (now - stored_at).total_seconds() > FRESHNESS_HOURS * 3600:
                        stale_count += 1
                except (ValueError, KeyError):
                    pass

        # If more than half the chunks are stale, flag it
        if stale_count > len(metadatas) / 2:
            freshness_ok = False

    # ── Coverage Check (LLM-based) ────────────────────────────────────────
    coverage_pass = False
    if similarity_pass:
        if best_score >= 0.60:
            emit_event(session_id, "grade", f"Fast-Path Coverage: Score {best_score:.2f} >= 0.60. Skipping LLM judge.")
            coverage_pass = True
        else:
            prompt = PromptTemplate(
                template=(
                    "You are the Context Awareness Gate. Assess whether the retrieved context "
                    "adequately covers what the user needs.\n"
                    "Context:\n{context}\n\nUser Query: {question}\n"
                    "Does this context provide strong, specific coverage of the user's question? Answer 'yes' or 'no'."
                ),
                input_variables=["context", "question"],
            )
            chain = prompt | structured_llm
            
            # Deduplicate chunks to avoid passing identical chunks if there are duplicates
            unique_docs = list(dict.fromkeys(documents))
            # Combine the chunks to see if collectively they answer the question
            combined_context = "\n\n---\n\n".join(unique_docs)
            
            try:
                result = chain.invoke({"context": combined_context, "question": question})
                if result.score.lower() == "yes":
                    coverage_pass = True
            except Exception as e:
                print(f"Coverage check failed: {e}")

    context_strong = similarity_pass and coverage_pass and freshness_ok

    if context_strong:
        arc.adjust_on_strong_context()
        emit_event(session_id, "grade", "Context STRONG — routing to Context Builder")
    else:
        arc.adjust_on_weak_context()
        reasons = []
        if not similarity_pass:
            reasons.append("low similarity")
        if not coverage_pass:
            reasons.append("low coverage")
        if not freshness_ok:
            reasons.append("stale chunks")
        emit_event(session_id, "grade", f"Context WEAK ({', '.join(reasons)}) — routing to External Knowledge")

    return {
        "web_fallback": not context_strong,
        "freshness_ok": freshness_ok,
        "original_documents": state.get("documents", []),  # store existing chunks to prevent deletion by web_search
        "pipeline_steps": steps + ["grade"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3a — External Knowledge Source + Dynamic Chunking Module
# ─────────────────────────────────────────────────────────────────────────────
def web_search_node(state: AgentState):
    """
    External Knowledge Source → Dynamic Chunking Module
    Uses ARC-driven chunk_size + chunk_overlap for adaptive chunking.
    Dynamic Chunking features:
      • Adaptive Chunk Size   — from arc.chunk_size
      • Semantic Split        — sentence-aware separators
      • Overlap Tuning        — from arc.chunk_overlap
    """
    question = state["question"]
    session_id = state.get("arc_params", {}).get("_session_id", "")
    steps = list(state.get("pipeline_steps", []))

    emit_event(session_id, "web_search", f"Querying External Knowledge Source via Tavily")

    docs = search_tool.invoke({"query": question})

    if isinstance(docs, dict):
        if "results" in docs and isinstance(docs["results"], list):
            docs = docs["results"]  # Extract the list of results
        else:
            docs = [docs]  # Wrap in list so it can be handled below

    if isinstance(docs, str):
        raw_text = docs
    elif isinstance(docs, list):
        parts = []
        for d in docs:
            if isinstance(d, dict):
                title = d.get("title", "")
                content = d.get("content", str(d))
                parts.append(f"[{title}]\n{content}" if title else content)
            else:
                parts.append(str(d))
        raw_text = "\n\n".join(parts)
    else:
        raw_text = str(docs)

    # Dynamic Chunking Module — ARC-adaptive
    # Read chunk params from the per-request snapshot set by retrieve_node
    # (which already called arc.adjust_chunk_size and re-snapshotted).
    # This ensures concurrent requests use their own query-tuned values.
    _snap        = state.get("arc_params", {})
    chunk_size   = _snap.get("chunk_size",   arc.chunk_size)
    chunk_overlap = _snap.get("chunk_overlap", arc.chunk_overlap)
    emit_event(
        session_id,
        "chunk",
        f"Dynamic Chunking: size={chunk_size}, overlap={chunk_overlap}, semantic split"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    chunks = text_splitter.split_text(raw_text)

    emit_event(session_id, "embed", f"Embedding {len(chunks)} chunks via Embedding Model")

    return {"documents": chunks, "pipeline_steps": steps + ["web_search", "chunk", "embed"]}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3b — Credibility Scoring + Memory Update (Vector Store)
# ─────────────────────────────────────────────────────────────────────────────
def credibility_node(state: AgentState):
    """
    Credibility Scoring → Memory Update (Vector Store)
    Only chunks that pass credibility scoring are written into long-term memory.
    """
    question = state["question"]
    chunks = state.get("documents", [])
    session_id = state.get("arc_params", {}).get("_session_id", "")
    steps = list(state.get("pipeline_steps", []))

    emit_event(session_id, "credibility", f"Credibility Scoring {len(chunks)} web chunks")

    if not chunks:
        return {"pipeline_steps": steps + ["credibility"]}

    prompt = PromptTemplate(
        template=(
            "You are a Credibility Scorer. Evaluate this chunk for factual accuracy "
            "and direct relevance to the user query.\n"
            "Chunk: {chunk}\nQuestion: {question}\n"
            "Should this chunk be trusted and saved to long-term Vector Memory? "
            "Answer 'yes' or 'no'."
        ),
        input_variables=["chunk", "question"],
    )
    chain = prompt | structured_llm

    credible_chunks = []

    # Run LLM credibility checks in parallel for all chunks
    inputs = [{"chunk": chunk, "question": question} for chunk in chunks]
    results = chain.batch(inputs)

    for chunk, result in zip(chunks, results):
        if result.score.lower() == "yes":
            credible_chunks.append(chunk)

    if credible_chunks:
        for chunk in credible_chunks:
            doc_id = str(uuid.uuid4())
            add_document_to_vector_store(
                doc_id, chunk,
                metadata={"source": "web_search_credible", "question": question}
            )

    emit_event(
        session_id,
        "memory_update",
        f"Memory Update: {len(credible_chunks)}/{len(chunks)} credible chunks stored"
    )

    # Fall back to all chunks if none passed credibility (don't leave context empty)
    return {
        "documents": credible_chunks if credible_chunks else chunks,
        "pipeline_steps": steps + ["credibility", "memory_update"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 — Context Builder + Generator Model (LLM)
# ─────────────────────────────────────────────────────────────────────────────
class MedResponse(BaseModel):
    answer: str = Field(description="The detailed medical answer to the user's query.")
    hallucination_flagged: bool = Field(description="True ONLY if the answer contains medical claims not supported by the context or general medical knowledge.")

def generate_node(state: AgentState):
    """
    Context Builder → Generator Model (LLM)
    Assembles ranked context and generates the answer in a single structured output, replacing the Critic node.
    """
    documents = state.get("documents", [])
    metadatas = state.get("metadatas", [])
    messages = state.get("messages", [])
    session_id = state.get("arc_params", {}).get("_session_id", "")
    steps = list(state.get("pipeline_steps", []))

    emit_event(session_id, "context_builder", "Building ranked context from retrieved chunks")

    # Context Builder — prepend metadata to each chunk for better grounding
    context_chunks = []
    seen = set()
    for i, doc in enumerate(documents if isinstance(documents, list) else [documents]):
        if doc in seen:
            continue
        seen.add(doc)
        meta_str = ""
        if isinstance(metadatas, list) and i < len(metadatas) and metadatas[i]:
            meta = metadatas[i]
            source = meta.get("filename", meta.get("source", "Unknown"))
            meta_str = f"[Source: {source}]\n"
        context_chunks.append(f"{meta_str}{doc}")

    context = "\n\n---\n\n".join(context_chunks)

    emit_event(session_id, "generate", "Generator Model (GPT-4o-mini) synthesizing answer and validating")

    system_prompt = (
        "You are **MedAI**, an expert clinical knowledge assistant built on an Adaptive Context-Aware RAG pipeline.\n"
        "RESPONSE RULES:\n"
        "1. **Direct and Specific First**: Lead your response with the direct, specific answer. State the exact specific term, protocol, or numeric range requested immediately.\n"
        "2. **Precision over Verbosity**: Do not expand with generic background info unless asked. Bullet points are fine but keep the core answer at the very top.\n"
        "3. **Context Grounding**: Base your answer on the retrieved clinical context below.\n"
        "4. **No Hallucinations**: Do NOT apply clinical facts from one disease/drug/condition to another.\n"
        "5. **Empty Context**: If the context below does NOT contain the answer, use your general parametric medical knowledge to answer, but you MUST prefix your answer with EXACTLY: '*(From general medical knowledge, not found in retrieved documents)*\\n\\n'.\n"
        "6. **Disclaimer**: Always end responses involving symptoms, diagnoses, or treatments with: "
        "'⚕️ **Medical Disclaimer**: This information is for educational purposes only. Always consult a licensed healthcare professional before making any medical decisions.'\n\n"
        f"Clinical Context:\n{context}"
    )

    input_messages = [SystemMessage(content=system_prompt)] + list(messages)
    
    chain = llm.with_structured_output(MedResponse)
    response = chain.invoke(input_messages)
    
    final_ans = response.answer
    hallucination_flagged = response.hallucination_flagged
    
    if hallucination_flagged:
        final_ans += "\n\n> ⚠️ **Validator Warning**: This response was flagged for potential hallucinations or unsupported claims. Please verify independently."

    emit_event(session_id, "done", "Final output ready")

    return {
        "final_answer": final_ans,
        "hallucination_flagged": hallucination_flagged,
        "needs_feedback": True,
        "pipeline_steps": steps + ["context_builder", "generate", "done"],
    }



# Build LangChain LCEL Pipeline
# ─────────────────────────────────────────────────────────────────────────────
#
# Topology (mirrors the original LangGraph flow):
#
#   retrieve → grade ──► [web_fallback=True]  → web_search → credibility → generate → critic
#                    └──► [web_fallback=False] → generate → critic
#
# Implementation:
#   • Each node is wrapped in _wrap_node() which merges partial returns into state.
#   • RunnableBranch handles the conditional routing after grade.
#   • The full sequence is: retrieve → grade → branch → critic
# ─────────────────────────────────────────────────────────────────────────────

retrieve_r    = _wrap_node(retrieve_node)
grade_r       = _wrap_node(grade_documents_node)
web_search_r  = _wrap_node(web_search_node)
credibility_r = _wrap_node(credibility_node)
generate_r    = _wrap_node(generate_node)

# Web path: web_search → credibility → generate
_web_path = web_search_r | credibility_r | generate_r

# Conditional branch — mirrors decide_to_generate()
_branch = RunnableBranch(
    (lambda state: state.get("web_fallback", True), _web_path),
    generate_r,  # default: context strong, go straight to generate
)

# Complete LCEL pipeline
pipeline = retrieve_r | grade_r | _branch


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
async def process_message(session_id: str, message: str) -> dict:
    # Restore or start conversation history from in-memory session store
    messages = list(_session_memory.get(session_id, []))
    messages.append(HumanMessage(content=message))

    inputs: AgentState = {
        "question": message,
        "messages": messages,
        "arc_params": arc.get_params() | {"_session_id": session_id},
        "pipeline_steps": [],
        "documents": [],
        "original_documents": [],
        "metadatas": [],
        "distances": [],
        "web_fallback": False,
        "freshness_ok": True,
        "needs_feedback": False,
        "final_answer": "",
    }

    # Run the LCEL pipeline (synchronous invoke inside async context)
    result = await asyncio.get_event_loop().run_in_executor(
        None, pipeline.invoke, inputs
    )

    # Persist AI response to in-memory session memory
    final_answer = result.get("final_answer", "")
    if final_answer:
        messages.append(AIMessage(content=final_answer))
    _session_memory[session_id] = messages

    return {
        "content": final_answer,
        "needs_feedback": result.get("needs_feedback", False),
        "thread_id": session_id,
        "pipeline_steps": result.get("pipeline_steps", []),
        "arc_params": result.get("arc_params", arc.get_params()),
        "web_fallback": result.get("web_fallback", False),
    }
