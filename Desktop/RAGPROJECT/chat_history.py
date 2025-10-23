import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from custom_types import ChatSession, ChatMessage

class ChatHistoryManager:
    def __init__(self, storage_dir: str = "chat_sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
    
    def create_session(self, pdf_name: str) -> str:
        session_id = str(uuid.uuid4())
        session = ChatSession(
            session_id=session_id,
            pdf_name=pdf_name,
            created_at=datetime.now()
        )
        self._save_session(session)
        return session_id
    
    def add_message(self, session_id: str, question: str, answer: str, sources: List[str]):
        session = self._load_session(session_id)
        if session:
            message = ChatMessage(
                question=question,
                answer=answer,
                timestamp=datetime.now(),
                sources=sources
            )
            session.messages.append(message)
            self._save_session(session)
    
    def get_session_history(self, session_id: str) -> List[ChatMessage]:
        session = self._load_session(session_id)
        return session.messages if session else []
    
    def get_all_sessions(self) -> List[ChatSession]:
        sessions = []
        for file_path in self.storage_dir.glob("*.json"):
            session = self._load_session(file_path.stem)
            if session:
                sessions.append(session)
        return sorted(sessions, key=lambda x: x.created_at, reverse=True)
    
    def _save_session(self, session: ChatSession):
        file_path = self.storage_dir / f"{session.session_id}.json"
        with open(file_path, 'w') as f:
            json.dump(session.model_dump(), f, default=str, indent=2)
    
    def _load_session(self, session_id: str) -> Optional[ChatSession]:
        file_path = self.storage_dir / f"{session_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                data['messages'] = [
                    ChatMessage(
                        question=msg['question'],
                        answer=msg['answer'],
                        timestamp=datetime.fromisoformat(msg['timestamp']),
                        sources=msg['sources']
                    ) for msg in data['messages']
                ]
                return ChatSession(**data)
        return None
