from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional

class TranslationBase(BaseModel):
    source_text: str = Field(..., description="Original text to be translated")
    context: Optional[str] = Field(None, description="Optional context for translation")
    translation_type: str = Field(..., description="Type of translation: 'human_to_korpo' or 'korpo_to_human'")

class TranslationCreate(TranslationBase):
    pass

class TranslationRead(TranslationBase):
    id: int
    translated_text: str
    created_at: datetime
    user_id: int

    class Config:
        from_attributes = True

class TranslationResponse(BaseModel):
    translation: str
    state: str
    error_message: Optional[str] = None
