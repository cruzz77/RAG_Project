import pydantic 
from datetime import datetime
from typing import List, Optional

class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str] 
    source_id: str = None 

class RAGUpsertResult(pydantic.BaseModel):
     ingested: int 

class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]

class RAGQueryResult(pydantic.BaseModel):
    answer: str 
    sources: list[str]
    num_contexts: int

# 🆕 NEW: Chat History Model
class ChatMessage(pydantic.BaseModel):
    question: str
    answer: str
    timestamp: datetime
    sources: List[str]
    
class ChatSession(pydantic.BaseModel):
    session_id: str
    pdf_name: str
    messages: List[ChatMessage] = []
    created_at: datetime