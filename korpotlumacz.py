import os
from pathlib import Path
import logging
import json
from openai import OpenAI
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import asyncio

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')


class DialogProcessor:
    @staticmethod
    def extract_role_and_text(line: str) -> Tuple[str, str]:
        """Wydobywa rolę i tekst z linii dialogu"""
        parts = line.strip().split(']: ', 1)
        if len(parts) != 2:
            return None, None
        role = parts[0].strip('[')
        text = parts[1].strip()
        return role, text

    @staticmethod
    def find_translation_pairs(lines: List[str]) -> List[Dict]:
        """
        Znajduje pary tłumaczeń w dialogu, używając kontekstu
        Zwraca listę słowników z parami korpo-tłumaczenie i pełnym kontekstem
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
                    prev_role, prev_text = DialogProcessor.extract_role_and_text(
                        lines[j])
                    if prev_role == "Pracodawca":
                        pairs.append({
                            'korpo': prev_text,
                            'human': text,
                            'context': buffer.copy()
                        })
                        break

        return pairs


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
            logging.info(f"Initializing KorpoTlumacz with model: {model_name}")
            self.client = OpenAI(api_key=api_key)
            self.model_name = model_name
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.index = None
            self.examples: List[Dict] = []
            self.processor = DialogProcessor()
            logging.info("KorpoTlumacz initialized successfully")
        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            self.state = TranslatorState.ERROR
            self.error_message = str(e)
            raise

    def _set_state(self, state: str, error_message: str = None):
        self.state = state
        self.error_message = error_message

    async def load_from_directory(self, directory_path: str):
        """Wczytuje i przetwarza wszystkie pliki tekstowe z katalogu"""
        try:
            self._set_state(TranslatorState.LOADING)
            logging.info(f"Wczytuję pliki z katalogu: {directory_path}")

            all_examples = []
            for filename in os.listdir(directory_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(directory_path, filename)
                    logging.info(f"Przetwarzam plik: {filename}")

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()

                        conversation_pairs = DialogProcessor.find_translation_pairs(
                            lines)
                        all_examples.extend(conversation_pairs)

                        logging.info(
                            f"Znaleziono {len(conversation_pairs)} par tłumaczeń w pliku {filename}")
                    except Exception as e:
                        logging.error(f"Błąd podczas przetwarzania pliku {filename}: {str(e)}")

            self.examples.extend(all_examples)
            logging.info(f"Łącznie załadowano {len(all_examples)} par tłumaczeń")

            await self._update_index()
            self._set_state(TranslatorState.SUCCESS)

        except Exception as e:
            self._set_state(TranslatorState.ERROR, str(e))
            raise

    async def _update_index(self):
        """Aktualizuje index FAISS dla wszystkich przykładów"""
        if not self.examples:
            logging.warning("Brak przykładów do zaindeksowania")
            return

        texts = [ex['korpo'] for ex in self.examples]
        embeddings = self.embed_model.encode(texts)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

        logging.info(f"Zaktualizowano index z {len(texts)} przykładami")

    async def find_similar_examples(self, query: str, k: int = 3) -> List[Dict]:
        """Znajduje najbardziej podobne przykłady do zapytania"""
        if not self.index:
            logging.error("Index nie został zainicjalizowany")
            return []

        query_embedding = self.embed_model.encode([query])
        D, I = self.index.search(
            np.array(query_embedding).astype('float32'), k)

        similar_examples = []
        for idx in I[0]:
            if idx < len(self.examples):
                similar_examples.append(self.examples[idx])

        return similar_examples

    async def _translate_to_human_internal(self, korpo_text: str, context: str = "") -> str:
        similar = await self.find_similar_examples(korpo_text)

        examples_text = "\n\n".join([
            f"Kontekst rozmowy:\n" + "\n".join(ex['context']) +
            f"\nKorpomowa: {ex['korpo']}\nTłumaczenie: {ex['human']}"
            for ex in similar
        ])

        context_info = f"\nKontekst obecnej sytuacji: {context}" if context else ""

        prompt = f"""Przetłumacz korpomowę na prosty język ludzki.
        Bądź bezpośredni i szczery jak korpotłumacz w przykładach.
        {context_info}

        Przykłady:
        {examples_text}

        Korpomowa do przetłumaczenia: {korpo_text}

        Tłumaczenie (uwzględniając podany kontekst):"""

        response = await asyncio.to_thread(
            lambda: self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Jesteś korpotłumaczem, który tłumaczy korporacyjną nowomowę na prosty język ludzki. Twoje tłumaczenia są bezkompromisowe i pokazują prawdziwą intencję wypowiedzi."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
        )

        return response.choices[0].message.content.strip()

    async def translate_to_human(self, korpo_text: str, context: str = "") -> Dict:
        """Asynchronous translation with state handling"""
        try:
            self._set_state(TranslatorState.LOADING)

            result = await self._translate_to_human_internal(korpo_text, context)

            self._set_state(TranslatorState.SUCCESS)
            return {
                "translation": result,
                "state": self.state,
                "original": korpo_text,
                "context": context
            }
        except Exception as e:
            self._set_state(TranslatorState.ERROR, str(e))
            raise

    async def _translate_to_korpo_internal(self, human_text: str, context: str = "") -> str:
        similar = await self.find_similar_examples(human_text)

        examples_text = "\n\n".join([
            f"Kontekst rozmowy:\n" + "\n".join(ex['context']) +
            f"\nLudzki język: {ex['human']}\nKorpomowa: {ex['korpo']}"
            for ex in similar
        ])

        context_info = f"\nKontekst obecnej sytuacji: {context}" if context else ""

        prompt = f"""Przetłumacz prosty tekst na korpomowę.
        Używaj profesjonalnego, korporacyjnego języka.
        {context_info}

        Przykłady:
        {examples_text}

        Tekst do przetłumaczenia: {human_text}

        Korpomowa (uwzględniając podany kontekst):"""

        response = await asyncio.to_thread(
            lambda: self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Jesteś korpotłumaczem, który przekształca proste wypowiedzi w profesjonalną korpomowę."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
        )

        return response.choices[0].message.content.strip()

    async def translate_to_korpo(self, human_text: str, context: str = "") -> Dict:
        """Asynchronous translation with state handling"""
        try:
            self._set_state(TranslatorState.LOADING)

            result = await self._translate_to_korpo_internal(human_text, context)

            self._set_state(TranslatorState.SUCCESS)
            return {
                "translation": result,
                "state": self.state,
                "original": human_text,
                "context": context
            }
        except Exception as e:
            self._set_state(TranslatorState.ERROR, str(e))
            raise

    async def save_examples(self, file_path: str):
        """Zapisuje bazę przykładów do pliku"""
        try:
            self._set_state(TranslatorState.LOADING)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.examples, f, ensure_ascii=False, indent=2)
            logging.info(f"Zapisano {len(self.examples)} przykładów do {file_path}")
            self._set_state(TranslatorState.SUCCESS)
        except Exception as e:
            self._set_state(TranslatorState.ERROR, str(e))
            raise

    async def load_examples(self, file_path: str):
        """Wczytuje przykłady z pliku"""
        try:
            self._set_state(TranslatorState.LOADING)
            with open(file_path, 'r', encoding='utf-8') as f:
                self.examples = json.load(f)
            await self._update_index()
            logging.info(f"Wczytano {len(self.examples)} przykładów z {file_path}")
            self._set_state(TranslatorState.SUCCESS)
        except Exception as e:
            self._set_state(TranslatorState.ERROR, str(e))
            raise
