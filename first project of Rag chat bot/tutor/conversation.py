"""
Conversation state and memory manager for PDF Tutor AI.
Maintains chat history, citations, and conversation context.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    mode: str
    citations: List[int] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    is_retrieval_only: bool = False
    timestamp: float = field(default_factory=time.time)


class ConversationManager:
    """
    Manages chat history and extracts recent context for follow-up questions.
    """
    
    def __init__(self):
        self.messages: List[ChatMessage] = []

    def add_user_message(self, content: str, mode: str) -> ChatMessage:
        msg = ChatMessage(
            role="user",
            content=content,
            mode=mode,
        )
        self.messages.append(msg)
        return msg

    def add_assistant_message(
        self,
        content: str,
        retrieved_chunks: List[Dict[str, Any]],
        mode: str,
        is_retrieval_only: bool = False,
    ) -> ChatMessage:
        # Extract unique, sorted page numbers
        citations = sorted(
            list(
                set(
                    chunk.get("page_number")
                    for chunk in retrieved_chunks
                    if chunk.get("page_number") is not None
                )
            )
        )
        
        msg = ChatMessage(
            role="assistant",
            content=content,
            mode=mode,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            is_retrieval_only=is_retrieval_only,
        )
        self.messages.append(msg)
        return msg

    def get_recent_context(self, max_turns: int = 3) -> str:
        """
        Extract the last few turns of user-assistant dialogue to provide
        continuity for follow-up questions without flooding the prompt.
        """
        if not self.messages:
            return ""
            
        recent = self.messages[-(max_turns * 2):]
        context_lines = []
        for m in recent:
            role_label = "Student" if m.role == "user" else "Tutor"
            # Truncate very long responses in memory context
            snippet = m.content[:300] + "..." if len(m.content) > 300 else m.content
            context_lines.append(f"{role_label}: {snippet}")
            
        return "\n".join(context_lines)

    def clear(self) -> None:
        """Reset conversation history."""
        self.messages.clear()
