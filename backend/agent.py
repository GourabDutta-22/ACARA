"""
Adaptive Context-Aware Retrieval Agent (LangGraph)
===================================================
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
"""

import os
import uuid
import asyncio
from datetime import datetime, timezone, timedelta
from typing import TypedDict, Annotated, Sequence, Optional, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database import search_vector_store, add_document_to_vector_store, USE_PINECONE, embeddings_model
from arc import arc
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Agent State
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
# NODE 1 — Query Encoder + Vector Memory Retrieval
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_node(state: AgentState):
    """
    User Query → Query Encoder → Vector Memory
    Uses ARC.top_k for dynamic number of chunks retrieved.
    """
    question = state["question"]
    session_id = state.get("arc_params", {}).get("_session_id", "")
    emit_event(session_id, "retrieve", f"Encoding query and searching Vector Memory (top-k={arc.top_k})")

    # Adjust chunk size heuristically before querying
    arc.adjust_chunk_size(question)

    # Generate embedding if using Pinecone or for consistent grading
    query_embedding = None
    if USE_PINECONE:
        query_embedding = embeddings_model.embed_query(question)

    results = search_vector_store(question, n_results=arc.top_k, embedding=query_embedding)
    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []

    return {
        "documents": documents,
        "metadatas": metadatas,
        "distances": distances,
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
    threshold = arc.similarity_threshold
    
    similarity_pass = False
    if distances:
        if USE_PINECONE:
            # Pinecone: distance = 1 - cosine_score. We want cosine_score >= threshold
            similarity_pass = any((1 - d) >= threshold for d in distances)
        else:
            # ChromaDB L2 distance: lower = more similar
            similarity_pass = any((1 / (1 + d)) >= threshold for d in distances)

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
        prompt = PromptTemplate(
            template=(
                "You are the Context Awareness Gate. Assess whether the retrieved document "
                "adequately covers what the user needs.\n"
                "Document: {document}\nUser Query: {question}\n"
                "Does this document provide strong, specific coverage? Answer 'yes' or 'no'."
            ),
            input_variables=["document", "question"],
        )
        chain = prompt | structured_llm
        for doc in documents[:2]:  # check top 2 docs only
            result = chain.invoke({"document": doc, "question": question})
            if result.score == "yes":
                coverage_pass = True
                break

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
        "original_documents": state.get("documents", []), # store existing chunks to prevent deletion by web_search
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
    chunk_size = arc.chunk_size
    chunk_overlap = arc.chunk_overlap
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
    
    # Batch embed credible chunks to be efficient
    temp_list = []
    
    for chunk in chunks:
        result = chain.invoke({"chunk": chunk, "question": question})
        if result.score == "yes":
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
def generate_node(state: AgentState):
    """
    Context Builder → Generator Model (LLM)
    Assembles ranked context and generates the answer.
    """
    documents = state.get("documents", [])
    metadatas = state.get("metadatas", [])
    messages = state.get("messages", [])
    session_id = state.get("arc_params", {}).get("_session_id", "")
    steps = list(state.get("pipeline_steps", []))

    emit_event(session_id, "context_builder", "Building ranked context from retrieved chunks")

    # Context Builder — prepend metadata to each chunk for better grounding
    context_chunks = []
    for i, doc in enumerate(documents if isinstance(documents, list) else [documents]):
        meta_str = ""
        if isinstance(metadatas, list) and i < len(metadatas) and metadatas[i]:
            meta = metadatas[i]
            source = meta.get("filename", meta.get("source", "Unknown"))
            meta_str = f"[Source: {source}]\n"
        context_chunks.append(f"{meta_str}{doc}")

    context = "\n\n---\n\n".join(context_chunks)

    emit_event(session_id, "generate", "Generator Model (GPT-4o-mini) synthesizing answer")

    system_prompt = (
        "You are a highly capable AI assistant part of an Adaptive Context-Aware RAG system. "
        "Answer the user's question strictly based on the retrieved context below. "
        "CRITICAL RULE: If the user asks about a specific person, entity, or object by name, you MUST verify that the name "
        "actually appears in the context in relation to the requested information. Do NOT apply facts from one person/entity "
        "to another person/entity. If the requested information for the specific person/entity is not in the context, "
        "clearly state that you don't have that information. Do not hallucinate.\n\n"
        f"Context:\n{context}"
    )

    input_messages = [SystemMessage(content=system_prompt)] + list(messages)
    response = llm.invoke(input_messages)

    return {
        "final_answer": response.content,
        "pipeline_steps": steps + ["context_builder", "generate"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5 — Critic / Validator
# ─────────────────────────────────────────────────────────────────────────────
def critic_node(state: AgentState):
    """
    Critic / Validator → Final Output
    Flags hallucinations or unsupported claims.
    """
    question = state["question"]
    documents = state.get("documents", [])
    answer = state["final_answer"]
    session_id = state.get("arc_params", {}).get("_session_id", "")
    steps = list(state.get("pipeline_steps", []))

    emit_event(session_id, "critic", "Critic / Validator checking for hallucinations")

    context = "\n\n---\n\n".join(documents) if isinstance(documents, list) else str(documents)

    prompt = PromptTemplate(
        template=(
            "You are a Critic Validator in an AI pipeline. Evaluate whether the answer "
            "hallucinates or contains claims completely unsupported by the context.\n"
            "Context: {context}\nQuestion: {question}\nAnswer: {answer}\n"
            "Does the answer hallucinate? Answer 'yes' if it does, 'no' if it is clean."
        ),
        input_variables=["context", "question", "answer"],
    )
    chain = prompt | structured_llm
    result = chain.invoke({"context": context, "question": question, "answer": answer})

    final_output = answer
    if result.score == "yes":
        final_output = (
            answer
            + "\n\n> ⚠️ **Validator Warning**: This response was flagged for potential "
            "hallucinations or unsupported claims. Please verify independently."
        )

    emit_event(session_id, "done", "Final output ready")

    return {
        "final_answer": final_output,
        "needs_feedback": True,
        "pipeline_steps": steps + ["critic", "done"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Edge
# ─────────────────────────────────────────────────────────────────────────────
def decide_to_generate(state: AgentState):
    """Context Awareness Gate routing."""
    return "web_search" if state.get("web_fallback", True) else "generate"


# ─────────────────────────────────────────────────────────────────────────────
# Build LangGraph
# ─────────────────────────────────────────────────────────────────────────────
from langgraph.checkpoint.memory import MemorySaver

workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("credibility", credibility_node)
workflow.add_node("generate", generate_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade",
    decide_to_generate,
    {"web_search": "web_search", "generate": "generate"},
)
workflow.add_edge("web_search", "credibility")
workflow.add_edge("credibility", "generate")
workflow.add_edge("generate", "critic")
workflow.add_edge("critic", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
async def process_message(session_id: str, message: str) -> dict:
    config = {"configurable": {"thread_id": session_id}}

    # Restore or start conversation history
    current_state = app.get_state(config)
    messages = (
        current_state.values.get("messages", [])
        if getattr(current_state, "values", None)
        else []
    )
    messages = list(messages)
    messages.append(HumanMessage(content=message))

    inputs = {
        "question": message,
        "messages": messages,
        "arc_params": arc.get_params() | {"_session_id": session_id},
        "pipeline_steps": [],
        "documents": [],
        "metadatas": [],
        "distances": [],
        "web_fallback": False,
        "freshness_ok": True,
    }

    result = app.invoke(inputs, config=config)

    # Append AI response to conversation history persisted in graph memory
    final_answer = result.get("final_answer", "")
    updated_messages = list(result.get("messages", messages))
    if final_answer:
        updated_messages.append(AIMessage(content=final_answer))

    return {
        "content": final_answer,
        "needs_feedback": result.get("needs_feedback", False),
        "thread_id": session_id,
        "pipeline_steps": result.get("pipeline_steps", []),
        "arc_params": result.get("arc_params", arc.get_params()),
        "web_fallback": result.get("web_fallback", False),
    }
