from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from app.models.translation import Translation
from app.schemas.translation import TranslationCreate

def create_translation(db: Session, translation: TranslationCreate, user_id: int, translated_text: str) -> Translation:
    db_translation = Translation(
        source_text=translation.source_text,
        translated_text=translated_text,
        context=translation.context,
        translation_type=translation.translation_type,
        user_id=user_id
    )
    db.add(db_translation)
    db.commit()
    db.refresh(db_translation)
    return db_translation

def get_translation(db: Session, translation_id: int) -> Optional[Translation]:
    return db.query(Translation).filter(Translation.id == translation_id).first()

def get_translations_by_user(
    db: Session, 
    user_id: int, 
    skip: int = 0, 
    limit: int = 10
) -> List[Translation]:
    return (
        db.query(Translation)
        .filter(Translation.user_id == user_id)
        .order_by(desc(Translation.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )

def get_user_translation_count(db: Session, user_id: int) -> int:
    return db.query(Translation).filter(Translation.user_id == user_id).count()
