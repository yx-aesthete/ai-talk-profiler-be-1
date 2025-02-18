from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "KorpoTlumacz API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str
    DATABASE_URL: str
    
    # JWT settings
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "https://corpotalk.lol",
        "https://www.corpotalk.lol",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "https://corpotalk-fe.onrender.com"
    ]
    
    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4"
    
    # Langfuse
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    
    # Rate limiting
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour
    RATE_LIMIT_MAX_REQUESTS: int = 100  # requests per window
    
    # Translator settings
    TRANSLATOR_CACHE_TIMEOUT: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
