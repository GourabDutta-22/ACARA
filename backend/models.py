from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    needs_feedback: bool
    thread_id: str
    pipeline_steps: List[str] = []
    arc_params: Dict[str, Any] = {}
    web_fallback: bool = False


class FeedbackRequest(BaseModel):
    session_id: str
    thread_id: str
    is_helpful: bool
    comments: Optional[str] = None


class UploadRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None


class ARCStatusResponse(BaseModel):
    similarity_threshold: float
    top_k: int
    chunk_size: int
    chunk_overlap: int
    adjustment_count: int


class StatsResponse(BaseModel):
    document_count: int
    collection_name: str


class StreamChatRequest(BaseModel):
    session_id: str
    message: str
