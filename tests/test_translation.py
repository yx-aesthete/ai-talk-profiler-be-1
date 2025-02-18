import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.dependencies.database import get_db
from app.models.translation import Translation
from app.core.config import settings
import asyncio

# Test database URL
TEST_DATABASE_URL = "postgresql+asyncpg://test:test@localhost:5432/test_korpotlumacz"

# Create test database engine
engine = create_async_engine(TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def override_get_db():
    async with TestingSessionLocal() as session:
        yield session

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
async def test_db():
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Translation.metadata.create_all)
    
    # Run tests
    yield
    
    # Drop tables
    async with engine.begin() as conn:
        await conn.run_sync(Translation.metadata.drop_all)

@pytest.mark.asyncio
async def test_translate_to_human(test_client, test_db):
    # Test data
    test_data = {
        "source_text": "Zróbmy quick sync na EOD i domknijmy case.",
        "translation_type": "korpo_to_human",
        "context": "Rozmowa w biurze"
    }
    
    # Make request
    response = test_client.post(
        "/api/v1/translations/",
        json=test_data,
        headers={"X-API-Key": "test-api-key"}
    )
    
    # Assert response
    assert response.status_code == 200
    data = response.json()
    assert "translation" in data
    assert data["state"] == "success"
    assert data["error_message"] is None

@pytest.mark.asyncio
async def test_translate_to_korpo(test_client, test_db):
    # Test data
    test_data = {
        "source_text": "Spotkajmy się pod koniec dnia i zamknijmy ten temat.",
        "translation_type": "human_to_korpo",
        "context": "Rozmowa w biurze"
    }
    
    # Make request
    response = test_client.post(
        "/api/v1/translations/",
        json=test_data,
        headers={"X-API-Key": "test-api-key"}
    )
    
    # Assert response
    assert response.status_code == 200
    data = response.json()
    assert "translation" in data
    assert data["state"] == "success"
    assert data["error_message"] is None

@pytest.mark.asyncio
async def test_get_translation_history(test_client, test_db):
    # Make request
    response = test_client.get("/api/v1/translations/history")
    
    # Assert response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
