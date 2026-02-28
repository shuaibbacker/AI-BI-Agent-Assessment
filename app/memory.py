"""Conversational Memory System for the BI Agent."""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages conversational memory and analysis history."""
    
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.conversation_history: deque = deque(maxlen=max_history)
        self.analysis_cache: Dict[int, Dict] = {}
        self.board_metadata: Dict[int, Dict] = {}
        logger.info("MemoryManager initialized")
    
    def store_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Store a conversation message."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        self.conversation_history.append(entry)
        logger.debug(f"Stored message from {role}")
    
    def store_analysis(self, analysis_data: Dict):
        """Store analysis results for a board."""
        board_id = analysis_data.get("board_id")
        if board_id:
            self.analysis_cache[board_id] = {
                **analysis_data,
                "cached_at": datetime.now().isoformat()
            }
            logger.info(f"Stored analysis for board {board_id}")
    
    def store_board_metadata(self, board_id: int, metadata: Dict):
        """Store board metadata."""
        self.board_metadata[board_id] = {
            **metadata,
            "stored_at": datetime.now().isoformat()
        }
    
    def get_recent_analyses(self, limit: int = 5) -> List[Dict]:
        """Get recent board analyses."""
        # Sort by timestamp, most recent first
        analyses = sorted(
            self.analysis_cache.values(),
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        return analyses[:limit]
    
    def get_board_analysis(self, board_id: int) -> Optional[Dict]:
        """Get cached analysis for a specific board."""
        return self.analysis_cache.get(board_id)
    
    def get_conversation_context(self, last_n: int = 5) -> List[Dict]:
        """Get last N conversation entries."""
        return list(self.conversation_history)[-last_n:]
    
    def is_board_known(self, board_id: int) -> bool:
        """Check if we've analyzed this board before."""
        return board_id in self.analysis_cache or board_id in self.board_metadata
    
    def get_board_summary(self, board_id: int) -> Optional[str]:
        """Get a text summary of what we know about a board."""
        analysis = self.analysis_cache.get(board_id)
        metadata = self.board_metadata.get(board_id)
        
        if not analysis and not metadata:
            return None
        
        parts = []
        if metadata:
            parts.append(f"Board '{metadata.get('board_name', 'Unknown')}' (ID: {board_id})")
        
        if analysis:
            parts.append(f"Last analyzed: {analysis.get('timestamp', 'unknown')}")
            result = analysis.get("result", {})
            if "pipeline" in result:
                pipeline = result["pipeline"]
                parts.append(f"Pipeline value: ${pipeline.get('total_pipeline_value', 0):,.2f}")
            if "closed_revenue" in result:
                closed = result["closed_revenue"]
                won = closed.get("closed_won", {})
                parts.append(f"Closed won: ${won.get('revenue', 0):,.2f}")
        
        return " | ".join(parts)
    
    def clear(self):
        """Clear all memory."""
        self.conversation_history.clear()
        self.analysis_cache.clear()
        self.board_metadata.clear()
        logger.info("Memory cleared")
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "conversation_entries": len(self.conversation_history),
            "cached_analyses": len(self.analysis_cache),
            "known_boards": len(self.board_metadata),
            "max_history": self.max_history
        }
    
    def extract_board_references(self, query: str) -> List[int]:
        """Extract potential board IDs from a query."""
        import re
        # Look for numbers that might be board IDs
        numbers = re.findall(r'\b(\d{7,10})\b', query)
        return [int(n) for n in numbers]
    
    def get_relevant_context_for_query(self, query: str) -> Dict:
        """Get relevant memory context for a user query."""
        context = {
            "mentioned_boards": [],
            "recent_analyses": [],
            "conversation_history": []
        }
        
        # Check for board references
        board_ids = self.extract_board_references(query)
        for board_id in board_ids:
            if self.is_board_known(board_id):
                context["mentioned_boards"].append({
                    "board_id": board_id,
                    "summary": self.get_board_summary(board_id)
                })
        
        # Get recent analyses
        context["recent_analyses"] = self.get_recent_analyses(limit=3)
        
        # Get conversation history
        context["conversation_history"] = self.get_conversation_context(last_n=3)
        
        return context
    
    def format_context_for_prompt(self, context: Dict) -> str:
        """Format context into a string for the LLM prompt."""
        parts = []
        
        if context["mentioned_boards"]:
            parts.append("Previously analyzed boards mentioned in query:")
            for board in context["mentioned_boards"]:
                parts.append(f"  - {board['summary']}")
        
        if context["recent_analyses"] and not context["mentioned_boards"]:
            parts.append("Recently analyzed boards:")
            for analysis in context["recent_analyses"][:2]:
                board_id = analysis.get("board_id", "?")
                board_name = analysis.get("board_name", "Unknown")
                analysis_type = analysis.get("analysis_type", "analysis")
                parts.append(f"  - Board '{board_name}' (ID: {board_id}) - {analysis_type}")
        
        if parts:
            return "\n".join(parts)
        return ""


# Singleton instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create MemoryManager singleton."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
