"""
Session and conversation memory management.
"""
from typing import List, Dict, Any, Optional
import time

class ChatMessage:
    def __init__(self, role: str, content: str, citations: Optional[List[Dict[str, Any]]] = None):
        self.role = role  # user or assistant
        self.content = content
        self.citations = citations or []
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "citations": self.citations,
            "timestamp": self.timestamp
        }

class ConversationMemory:
    def __init__(self, max_history_turns: int = 10):
        self.max_history_turns = max_history_turns
        self.messages: List[ChatMessage] = []

    def add_user_message(self, content: str):
        self.messages.append(ChatMessage(role="user", content=content))
        self._trim()

    def add_assistant_message(self, content: str, citations: Optional[List[Dict[str, Any]]] = None):
        self.messages.append(ChatMessage(role="assistant", content=content, citations=citations))
        self._trim()

    def _trim(self):
        max_msgs = self.max_history_turns * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]

    def get_history_context(self) -> str:
        lines = []
        for msg in self.messages:
            lines.append(f"{msg.role.upper()}: {msg.content}")
        return "\n".join(lines)

    def get_messages(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.messages]

    def clear(self):
        self.messages.clear()

class SessionMemoryManager:
    def __init__(self):
        self._sessions: Dict[str, ConversationMemory] = {}

    def get_session(self, session_id: str) -> ConversationMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory()
        return self._sessions[session_id]

    def clear_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]
