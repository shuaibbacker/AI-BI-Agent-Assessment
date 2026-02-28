"""Configuration settings for the BI Agent."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""
    
    # Monday.com API
    MONDAY_API_KEY: Optional[str] = os.getenv("MONDAY_API_KEY")
    MONDAY_API_URL: str = "https://api.monday.com/v2"
    
    # Groq API (primary)
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    # OpenAI API (fallback)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    
    # Gemini API (fallback)
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
    
    # API Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Memory
    MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "50"))
    
    # Data Cleaning
    DEFAULT_STAGE_PROBABILITIES: dict = {
        "lead": 0.10,
        "qualified": 0.25,
        "opportunity": 0.40,
        "proposal": 0.60,
        "closed_won": 1.0,
        "closed_lost": 0.0
    }
    
    @classmethod
    def validate(cls) -> list:
        """Validate required configuration. Returns list of missing settings."""
        missing = []
        
        if not cls.MONDAY_API_KEY:
            missing.append("MONDAY_API_KEY")
        
        if not cls.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        
        return missing
    
    @classmethod
    def is_valid(cls) -> bool:
        """Check if all required config is present."""
        return len(cls.validate()) == 0
