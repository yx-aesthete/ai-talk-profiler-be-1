from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from typing import Optional, List
from app.dependencies.database import get_db
from app.schemas.translation import TranslationCreate, TranslationRead, TranslationResponse
from app.crud.translation import create_translation, get_translations_by_user
from app.core.auth import current_active_user
from app.models.user import User
from app.utils.translator import get_translator, validate_api_key
from langfuse.decorators import observe
from app.core.config import settings

router = APIRouter(prefix="/api/v1/translations", tags=["translations"])

@router.post("/", response_model=TranslationResponse)
@observe(name="translate")
async def translate(
    translation: TranslationCreate,
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(current_active_user),
):
    if not x_api_key:
        raise HTTPException(status_code=400, detail="X-API-Key header is required")
    
    try:
        validate_api_key(x_api_key)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    translator = await get_translator(x_api_key)
    
    try:
        if translation.translation_type == "korpo_to_human":
            result = await translator.translate_to_human(
                translation.source_text,
                translation.context or ""
            )
        else:
            result = await translator.translate_to_korpo(
                translation.source_text,
                translation.context or ""
            )
        
        if translator.state == "success":
            # Save successful translation to database
            create_translation(
                db=db,
                translation=translation,
                user_id=current_user.id,
                translated_text=result
            )
        
        return TranslationResponse(
            translation=result,
            state=translator.state,
            error_message=translator.error_message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=List[TranslationRead])
async def get_translation_history(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(current_active_user),
):
    translations = get_translations_by_user(
        db=db,
        user_id=current_user.id,
        skip=skip,
        limit=limit
    )
    return translations
