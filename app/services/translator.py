from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import asyncio
import logging
from enum import Enum
import emoji

class TranslationState(str, Enum):
    IDLE = "idle" + " " + emoji.emojize(":zzz:")
    LOADING = "loading" + " " + emoji.emojize(":hourglass_flowing_sand:")
    ERROR = "error" + " " + emoji.emojize(":warning:")
    SUCCESS = "success" + " " + emoji.emojize(":check_mark_button:")

class DialogProcessor:
    @staticmethod
    def extract_role_and_text(line: str) -> Tuple[str, str]:
        """Extracts role and text from a dialog line 🎭"""
        parts = line.strip().split(']: ', 1)
        if len(parts) != 2:
            return None, None
        role = parts[0].strip('[')
        text = parts[1].strip()
        return role, text

    @staticmethod
    def find_translation_pairs(lines: List[str]) -> List[Dict]:
        """
        Finds translation pairs in dialog using context 🔍
        Returns a list of dictionaries with corpo-translation pairs and full context
        """
        pairs = []
        buffer = []

        for i, line in enumerate(lines):
            role, text = DialogProcessor.extract_role_and_text(line)
            if not role or not text:
                continue

            buffer.append(line)

            if role == "Korpotłumacz":
                for j in range(i-1, max(-1, i-5), -1):
                    prev_role, prev_text = DialogProcessor.extract_role_and_text(lines[j])
                    if prev_role == "Pracodawca":
                        pairs.append({
                            'korpo': prev_text,
                            'human': text,
                            'context': buffer.copy()
                        })
                        break

        return pairs

class TranslationService:
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        """Initialize translation service with OpenAI and embedding model 🚀"""
        self.state = TranslationState.IDLE
        self.error_message = None
        try:
            self.client = OpenAI(api_key=api_key)
            self.model_name = model_name
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.index = None
            self.examples: List[Dict] = []
            self.processor = DialogProcessor()
            logging.info(f"{emoji.emojize(':rocket:')} Translation service initialized successfully!")
        except Exception as e:
            self.state = TranslationState.ERROR
            self.error_message = str(e)
            logging.error(f"{emoji.emojize(':warning:')} Error initializing translation service: {e}")
            raise

    async def generate_translation_name(self, original_text: str, translation: str, context: str = "") -> str:
        """Generates a unique name for the translation based on content 🏷️"""
        try:
            prompt = f"""
            Create a short, unique name for the translation. The name should be concise (max few words)
            and should consider the original text, translation, and context.
            It can be humorous or creative, referencing the corporate speak and simple language style.

            Original text: "{original_text}"
            Translation: "{translation}"
            Context: "{context}"

            Translation name:
            """
            
            response = await asyncio.to_thread(
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=50
                )
            )
            
            name = response.choices[0].message.content.strip()
            logging.info(f"{emoji.emojize(':label:')} Generated translation name: {name}")
            return name
        except Exception as e:
            logging.error(f"{emoji.emojize(':warning:')} Error generating translation name: {e}")
            return f"Translation_{original_text[:20]}"

    async def translate_to_human(self, text: str, examples: Optional[List[Dict]] = None) -> Dict:
        """Translates corporate speak to human language 🔄"""
        self.state = TranslationState.LOADING
        try:
            context = self._prepare_context(examples) if examples else ""
            
            prompt = f"""
            {context}
            Translate the following corporate speak into simple Polish language.
            Make the translation clear, direct, and easy to understand.
            Maintain the meaning but remove corporate jargon.

            Corporate text: {text}

            Translation:"""

            response = await asyncio.to_thread(
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
            )

            translation = response.choices[0].message.content.strip()
            name = await self.generate_translation_name(text, translation)
            
            self.state = TranslationState.SUCCESS
            logging.info(f"{emoji.emojize(':white_check_mark:')} Successfully translated to human")
            
            return {
                "translation": translation,
                "name": name,
                "state": self.state,
                "error": None
            }
            
        except Exception as e:
            self.state = TranslationState.ERROR
            self.error_message = str(e)
            logging.error(f"{emoji.emojize(':warning:')} Translation error: {e}")
            return {
                "translation": None,
                "name": None,
                "state": self.state,
                "error": str(e)
            }

    async def translate_to_corpo(self, text: str, examples: Optional[List[Dict]] = None) -> Dict:
        """Translates human language to corporate speak 🔄"""
        self.state = TranslationState.LOADING
        try:
            context = self._prepare_context(examples) if examples else ""
            
            prompt = f"""
            {context}
            Translate the following simple Polish text into corporate speak.
            Use professional jargon, corporate buzzwords, and complex expressions.
            Maintain the core meaning while making it sound more "corporate".

            Simple text: {text}

            Translation:"""

            response = await asyncio.to_thread(
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
            )

            translation = response.choices[0].message.content.strip()
            name = await self.generate_translation_name(text, translation)
            
            self.state = TranslationState.SUCCESS
            logging.info(f"{emoji.emojize(':white_check_mark:')} Successfully translated to corpo")
            
            return {
                "translation": translation,
                "name": name,
                "state": self.state,
                "error": None
            }
            
        except Exception as e:
            self.state = TranslationState.ERROR
            self.error_message = str(e)
            logging.error(f"{emoji.emojize(':warning:')} Translation error: {e}")
            return {
                "translation": None,
                "name": None,
                "state": self.state,
                "error": str(e)
            }

    def _prepare_context(self, examples: List[Dict]) -> str:
        """Prepares context from translation examples 📚"""
        if not examples:
            return ""
            
        context = "Here are some example translations for reference:\n\n"
        for example in examples:
            context += f"Corporate: {example['korpo']}\n"
            context += f"Simple: {example['human']}\n\n"
        return context
