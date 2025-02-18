# KorpoTlumacz API

API backend for the KorpoTlumacz application, which translates between corporate jargon and simple Polish language.

## Features

- User authentication and authorization using FastAPI Users
- Translation between corporate jargon and simple Polish
- Translation history tracking
- Rate limiting
- Analytics integration with Langfuse
- OpenAI GPT-4 integration
- PostgreSQL database with SQLAlchemy

## Prerequisites

- Python 3.9+
- PostgreSQL
- OpenAI API key
- Langfuse account (for analytics)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/korpotlumacz-be.git
cd korpotlumacz-be
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the example environment file and update it with your settings:
```bash
cp .env.example .env
```

5. Update the following variables in `.env`:
- `SECRET_KEY`: Your application secret key
- `DATABASE_URL`: Your PostgreSQL connection string
- `OPENAI_API_KEY`: Your OpenAI API key
- `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`: Your Langfuse credentials

6. Initialize the database:
```bash
alembic upgrade head
```

## Running the Application

1. Start the development server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. Access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Authentication
- `POST /api/v1/auth/register`: Register a new user
- `POST /api/v1/auth/jwt/login`: Login and get JWT token
- `POST /api/v1/auth/jwt/logout`: Logout

### Translation
- `POST /api/v1/translations/`: Translate text
- `GET /api/v1/translations/history`: Get translation history

## Development

### Running Tests
```bash
pytest
```

### Database Migrations

Create a new migration:
```bash
alembic revision --autogenerate -m "description"
```

Apply migrations:
```bash
alembic upgrade head
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
