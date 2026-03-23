from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import asyncio
import json
import uuid
import uvicorn

load_dotenv()

from models import (
    ChatRequest, ChatResponse, FeedbackRequest,
    UploadRequest, ARCStatusResponse, StatsResponse, StreamChatRequest,
)
from database import feedback_collection, chat_collection, add_document_to_vector_store, get_collection_stats, USE_PINECONE
from arc import arc
from agent import process_message, get_or_create_queue, emit_event
from visualize_3d import generate_3d_viz

app = FastAPI(
    title="Adaptive Context-Aware RAG API",
    description="Full Adaptive Context-Aware Retrieval Architecture with ARC, Dynamic Chunking, Credibility Scoring, and Critic/Validator",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "Adaptive Context-Aware RAG Backend Running",
        "version": "2.0.0",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chat (standard JSON response)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        result = await process_message(request.session_id, request.message)

        # Persist both turns to MongoDB
        await chat_collection.insert_many([
            {"session_id": request.session_id, "role": "user", "content": request.message},
            {
                "session_id": request.session_id,
                "role": "ai",
                "content": result["content"],
                "web_fallback": result.get("web_fallback", False),
            },
        ])

        return ChatResponse(
            response=result["content"],
            needs_feedback=result.get("needs_feedback", False),
            thread_id=result.get("thread_id", str(uuid.uuid4())),
            pipeline_steps=result.get("pipeline_steps", []),
            arc_params=result.get("arc_params", {}),
            web_fallback=result.get("web_fallback", False),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Chat History — retrieve messages for a session
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """Return all messages for a given session, oldest first."""
    cursor = chat_collection.find(
        {"session_id": session_id},
        {"_id": 0, "role": 1, "content": 1, "web_fallback": 1}
    )
    messages = await cursor.to_list(length=500)
    return {"session_id": session_id, "messages": messages}


# ─────────────────────────────────────────────────────────────────────────────
# Sessions — list all unique session IDs with their first message as title
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/sessions")
async def get_sessions():
    """Return all sessions with a title derived from the first user message."""
    pipeline = [
        {"$sort": {"_id": 1}},
        {"$group": {
            "_id": "$session_id",
            "first_message": {"$first": "$content"},
            "role": {"$first": "$role"},
            "last_message_id": {"$last": "$_id"}
        }},
        {"$sort": {"last_message_id": -1}},
        {"$project": {"session_id": "$_id", "title": "$first_message", "_id": 0}},
    ]
    cursor = chat_collection.aggregate(pipeline)
    sessions = await cursor.to_list(length=200)
    # Only keep sessions where the first stored message was from a user
    sessions = [
        {"session_id": s["session_id"], "title": s["title"][:40] + ("…" if len(s["title"]) > 40 else "")}
        for s in sessions
    ]
    return {"sessions": sessions}


# ─────────────────────────────────────────────────────────────────────────────
# Delete Session
# ─────────────────────────────────────────────────────────────────────────────
@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Deletes all chat messages for a given session."""
    result = await chat_collection.delete_many({"session_id": session_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found or already deleted.")
    return {"status": "success", "deleted_count": result.deleted_count}


# ─────────────────────────────────────────────────────────────────────────────
# Stream Chat — SSE endpoint for real-time pipeline step events
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/stream-chat")
async def stream_chat_endpoint(request: StreamChatRequest):
    """
    Server-Sent Events endpoint.
    Emits pipeline step events as the agent processes the query,
    then sends the final answer as the last event.
    """
    session_id = request.session_id
    message = request.message

    # Register SSE queue for this session
    queue = get_or_create_queue(session_id)

    async def event_generator():
        # Run the agent in a background task
        agent_task = asyncio.create_task(
            process_message(session_id, message)
        )

        sent_done = False
        while not sent_done:
            try:
                # Drain all queued events (non-blocking)
                while True:
                    try:
                        event = queue.get_nowait()
                        data = json.dumps(event)
                        yield f"data: {data}\n\n"
                        if event.get("step") == "done":
                            sent_done = True
                    except asyncio.QueueEmpty:
                        break

                if agent_task.done():
                    # Drain remaining events
                    while not queue.empty():
                        try:
                            event = queue.get_nowait()
                            data = json.dumps(event)
                            yield f"data: {data}\n\n"
                        except asyncio.QueueEmpty:
                            break

                    # Send final answer
                    result = agent_task.result()
                    final_event = {
                        "step": "final",
                        "content": result.get("content", ""),
                        "needs_feedback": result.get("needs_feedback", False),
                        "pipeline_steps": result.get("pipeline_steps", []),
                        "arc_params": result.get("arc_params", {}),
                        "web_fallback": result.get("web_fallback", False),
                    }
                    yield f"data: {json.dumps(final_event)}\n\n"
                    break

                await asyncio.sleep(0.1)

            except Exception as e:
                error_event = {"step": "error", "detail": str(e)}
                yield f"data: {json.dumps(error_event)}\n\n"
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feedback
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    feedback_doc = request.model_dump()
    await feedback_collection.insert_one(feedback_doc)
    return {"status": "success", "message": "Feedback recorded."}


# ─────────────────────────────────────────────────────────────────────────────
# Upload — inject documents directly into Vector Memory
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/upload")
def upload_endpoint(request: UploadRequest, background_tasks: BackgroundTasks):
    """
    Accepts raw text + optional metadata and stores it in ChromaDB
    via the Dynamic Chunking pipeline (ARC-driven chunk size).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunk_size = arc.chunk_size
    chunk_overlap = arc.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(request.text)

    doc_ids = []
    for chunk in chunks:
        doc_id = str(uuid.uuid4())
        meta = request.metadata or {}
        add_document_to_vector_store(doc_id, chunk, metadata={**meta, "source": "manual_upload"})
        doc_ids.append(doc_id)

    background_tasks.add_task(generate_3d_viz)

    return {
        "status": "success",
        "chunks_stored": len(doc_ids),
        "doc_ids": doc_ids,
        "chunk_size_used": chunk_size,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Upload PDF — parse PDF and inject pages into Vector Memory
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/upload-pdf")
async def upload_pdf_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts a PDF file (multipart/form-data), extracts text via PyMuPDF,
    then stores ARC-driven chunks into ChromaDB.
    """
    import fitz  # PyMuPDF
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted at this endpoint.")

    # Read uploaded bytes
    pdf_bytes = await file.read()

    # Extract text page-by-page
    try:
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {e}")

    pages_text = []
    for page in pdf_doc:
        text = page.get_text("text").strip()
        if text:
            pages_text.append(text)
    pdf_doc.close()

    if not pages_text:
        raise HTTPException(status_code=422, detail="No extractable text found in PDF (may be a scanned image PDF).")

    full_text = "\n\n".join(pages_text)

    # ARC-driven chunking — same pipeline as /upload
    chunk_size = arc.chunk_size
    chunk_overlap = arc.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(full_text)

    doc_ids = []
    for chunk in chunks:
        doc_id = str(uuid.uuid4())
        add_document_to_vector_store(
            doc_id, chunk,
            metadata={"source": "pdf_upload", "filename": file.filename}
        )
        doc_ids.append(doc_id)

    background_tasks.add_task(generate_3d_viz)

    return {
        "status": "success",
        "filename": file.filename,
        "pages_extracted": len(pages_text),
        "chunks_stored": len(doc_ids),
        "chunk_size_used": chunk_size,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ARC Status and Reset
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/arc-status", response_model=ARCStatusResponse)
def arc_status():
    """Returns the current Adaptive Retrieval Controller parameters."""
    params = arc.get_params()
    return ARCStatusResponse(**params)


@app.post("/arc/reset")
def reset_arc():
    """Resets ARC to default starting parameters for a new chat session."""
    arc.reset()
    return {"status": "success", "message": "ARC parameters reset to defaults."}


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store Stats
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/stats", response_model=StatsResponse)
def stats_endpoint():
    """Returns basic stats about the Vector Memory (ChromaDB)."""
    stats = get_collection_stats()
    return StatsResponse(**stats)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
