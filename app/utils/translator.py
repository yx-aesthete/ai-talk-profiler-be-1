import os
from typing import Dict, Tuple
import time
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import logging
from app.core.config import settings

# Cache for translator instances
translator_instances: Dict[str, Tuple[KorpoTlumacz, float]] = {}
TRANSLATOR_CACHE_TIMEOUT = 3600  # 1 hour

class TranslatorState:
    IDLE = "idle"
    LOADING = "loading"
    ERROR = "error"
    SUCCESS = "success"

class KorpoTlumacz:
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        self.state = TranslatorState.IDLE
        self.error_message = None
        try:
            self.client = OpenAI(api_key=api_key)
            self.model_name = model_name
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.examples = []
            self._set_state(TranslatorState.IDLE)
        except Exception as e:
            self._set_state(TranslatorState.ERROR, str(e))
            raise

    def _set_state(self, state: str, error_message: str = None):
        self.state = state
        self.error_message = error_message

    async def translate_to_human(self, korpo_text: str, context: str = ""):
        self._set_state(TranslatorState.LOADING)
        try:
            result = await self._translate_to_human_internal(korpo_text, context)
            self._set_state(TranslatorState.SUCCESS)
            return result
        except Exception as e:
            self._set_state(TranslatorState.ERROR, str(e))
            raise

    async def _translate_to_human_internal(self, korpo_text: str, context: str = ""):
        messages = [
            {"role": "system", "content": "Jesteś pomocnym tłumaczem, który przekłada korpomowę na prosty język polski."},
            {"role": "user", "content": f"Przetłumacz następujący tekst z korpomowy na prosty język polski:\n\nTekst: {korpo_text}\n\nKontekst: {context if context else 'Brak dodatkowego kontekstu'}"}
        ]

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()

    async def translate_to_korpo(self, human_text: str, context: str = ""):
        self._set_state(TranslatorState.LOADING)
        try:
            result = await self._translate_to_korpo_internal(human_text, context)
            self._set_state(TranslatorState.SUCCESS)
            return result
        except Exception as e:
            self._set_state(TranslatorState.ERROR, str(e))
            raise

    async def _translate_to_korpo_internal(self, human_text: str, context: str = ""):
        messages = [
            {"role": "system", "content": "Jesteś pomocnym tłumaczem, który przekłada prosty język polski na korpomowę."},
            {"role": "user", "content": f"Przetłumacz następujący tekst z prostego języka polskiego na korpomowę:\n\nTekst: {human_text}\n\nKontekst: {context if context else 'Brak dodatkowego kontekstu'}"}
        ]

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()

def validate_api_key(api_key: str):
    if not api_key:
        raise ValueError("API key is required")
    if not api_key.startswith(("sk-", "test-")):
        raise ValueError("Invalid API key format")

async def get_translator(api_key: str) -> KorpoTlumacz:
    current_time = time.time()
    
    # Check if we have a cached instance
    if api_key in translator_instances:
        translator, timestamp = translator_instances[api_key]
        if current_time - timestamp < TRANSLATOR_CACHE_TIMEOUT:
            return translator
    
    # Create new instance
    translator = KorpoTlumacz(api_key=api_key)
    translator_instances[api_key] = (translator, current_time)
    
    return translator
