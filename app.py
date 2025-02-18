from quart import Quart, request, jsonify
from quart_cors import cors
from quart_rate_limiter import RateLimiter, rate_limit
from korpotlumacz import KorpoTlumacz, TranslatorState
import os
from functools import wraps
import logging
from typing import Dict
import time
from pathlib import Path
from langfuse import Langfuse
from langfuse.decorators import observe
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Langfuse
langfuse = Langfuse(
    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
    host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
)
from langfuse import Langfuse

# Podstawowa konfiguracja ścieżek
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "korpotlumacz_database.json"

langfuse = Langfuse(
  secret_key="sk-lf-950a84d6-eb83-43af-b5d2-b5c3381918f5",
  public_key="pk-lf-f91f8e2d-764b-4470-971f-a4e946c336af",
  host="https://cloud.langfuse.com"
)

app = Quart(__name__)
app = cors(app, 
    allow_origin=[
        "https://corpotalk.lol",
        "https://www.corpotalk.lol",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "https://corpotalk-fe.onrender.com"
    ],
    allow_headers=["Content-Type", "X-API-Key"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_credentials=True
)
limiter = RateLimiter(app)

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Logging for debugging
@app.before_request
async def log_request_info():
    logging.info(f"Incoming request: {request.method} {request.path}")
    logging.info(f"Headers: {request.headers}")
    body = await request.get_data()
    logging.info(f"Body: {body}")

# Cache dla instancji tłumacza
translator_instances: Dict[str, tuple[KorpoTlumacz, float]] = {}
TRANSLATOR_CACHE_TIMEOUT = 3600  # 1 godzina

def validate_api_key(api_key: str) -> bool:
    try:
        if not api_key.startswith('sk-'):
            return False
        return True
    except Exception:
        return False

async def get_translator(api_key: str) -> KorpoTlumacz:
    current_time = time.time()

    logging.info(f"Próba dostępu do tłumacza dla klucza API kończącego się na: ...{api_key[-4:]}")

    # Usuń przeterminowane instancje
    expired_keys = [
        key for key, (_, timestamp) in translator_instances.items()
        if current_time - timestamp > TRANSLATOR_CACHE_TIMEOUT
    ]
    for key in expired_keys:
        del translator_instances[key]

    if api_key in translator_instances:
        translator, _ = translator_instances[api_key]
        translator_instances[api_key] = (translator, current_time)
        return translator

    try:
        translator = KorpoTlumacz(api_key)
        await translator.load_examples(str(DATABASE_PATH))
        translator_instances[api_key] = (translator, current_time)
        return translator
    except Exception as e:
        logging.error(f"Błąd podczas tworzenia tłumacza: {str(e)}")
        raise

def require_api_key():
    def decorator(f):
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')

            if not api_key:
                return jsonify({
                    'status': 'error',
                    'message': 'Brak klucza API',
                    'code': 'missing_api_key'
                }), 401

            if not validate_api_key(api_key):
                return jsonify({
                    'status': 'error',
                    'message': 'Nieprawidłowy klucz API',
                    'code': 'invalid_api_key'
                }), 401

            return await f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/api/health')
async def health_check():
    return jsonify({
        'status': 'ok',
        'version': '1.0.0',
        'database_exists': os.path.exists(DATABASE_PATH),
        'active_translators': len(translator_instances)
    })

@app.route('/api/translate', methods=['POST'])
@require_api_key()
async def translate():
    try:
        data = await request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Brak danych w zapytaniu',
                'code': 'missing_data'
            }), 400

        required_fields = ['text', 'direction']
        if not all(field in data for field in required_fields):
            return jsonify({
                'status': 'error',
                'message': 'Brak wymaganych pól',
                'code': 'missing_fields'
            }), 400

        api_key = request.headers['X-API-Key']
        translator = await get_translator(api_key)

        text = data['text']
        direction = data['direction']
        context = data.get('context', '')

        if direction not in ['to_human', 'to_korpo']:
            return jsonify({
                'status': 'error',
                'message': 'Nieprawidłowy kierunek tłumaczenia',
                'code': 'invalid_direction'
            }), 400

        start_time = time.time()
        span = None
        
        try:
            # Create a trace for the translation request
            trace = langfuse.trace(name='translation_request')
            
            # Create span for timing
            span = trace.span(
                name=f'translate_{direction}',
                input={'text': text, 'context': context}
            )
            
            # Execute translation
            if direction == 'to_human':
                result = await translator.translate_to_human(text, context)
            else:
                result = await translator.translate_to_korpo(text, context)
                
            # Log success
            end_time = time.time()
            span.end(
                output=result,
                metadata={
                    'model': 'gpt-4',
                    'duration_ms': int((end_time - start_time) * 1000)
                }
            )

            return jsonify({
                'status': 'success',
                'data': result
            })

        except Exception as e:
            # Log error and end span with error status
            error_time = time.time()
            logging.error(f"Translation error: {str(e)}")
            if 'span' in locals():
                span.end(
                    error=str(e),
                    metadata={
                        'model': 'gpt-4',
                        'duration_ms': int((error_time - start_time) * 1000)
                    },
                    status='error'
                )
            return jsonify({
                'status': 'error',
                'message': 'Błąd podczas tłumaczenia',
                'code': 'translation_error',
                'details': str(e)
            }), 500

    except Exception as e:
        # Log server error
        error_time = time.time()
        logging.error(f"Server error: {str(e)}")
        
        # End span with error if it exists
        if 'span' in locals():
            span.end(
                error=str(e),
                metadata={
                    'error_type': 'server_error',
                    'duration_ms': int((error_time - start_time) * 1000) if 'start_time' in locals() else 0
                },
                status='error'
            )
            
        return jsonify({
            'status': 'error',
            'message': 'Błąd serwera',
            'code': 'server_error',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    import hypercorn.asyncio
    import asyncio

    config = hypercorn.Config()
    config.bind = ["0.0.0.0:443"]
    config.certfile = "/etc/letsencrypt/live/api.corpotalk.lol/fullchain.pem"
    config.keyfile = "/etc/letsencrypt/live/api.corpotalk.lol/privkey.pem"
    config.alpn_protocols = ["h2", "http/1.1"]
    config.verify_mode = None

    asyncio.run(hypercorn.asyncio.serve(app, config))