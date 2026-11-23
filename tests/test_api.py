"""API endpoint tests using FastAPI TestClient."""
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create test client fixture."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test GET /health returns HTTP 200 and {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_request(client):
    """Test POST /predict with valid payload returns HTTP 200 and prediction value."""
    payload = {
        "students_registered": 950,
        "attendance_pct": 75.0,
        "menu_rating": 4.2,
        "temp_c": 28.5,
        "cooked_kg": 320.0,
        "plate_size_cm": 27.5,
        "meal": "lunch",
        "hostel": "Boys Hostel 1"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))
    assert data["prediction"] >= 0.0
    assert data.get("status") == "success"


def test_predict_rejects_excess_students(client):
    """Test Rule 1: students_registered > 2000 is rejected with HTTP 422."""
    payload = {
        "students_registered": 2500,  # Invalid: exceeds 2000
        "attendance_pct": 80.0,
        "menu_rating": 3.0,
        "temp_c": 25.0,
        "cooked_kg": 300.0,
        "plate_size_cm": 26.0,
        "meal": "lunch",
        "hostel": "Girls Hostel"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    errors = response.json().get("detail", [])
    assert any("students_registered" in str(err) for err in errors)


def test_predict_rejects_sub_minimum_cooked_kg(client):
    """Test Rule 2: cooked_kg < 25 is rejected with HTTP 422."""
    payload = {
        "students_registered": 500,
        "attendance_pct": 60.0,
        "menu_rating": 3.5,
        "temp_c": 26.0,
        "cooked_kg": 15.0,  # Invalid: below 25 kg
        "plate_size_cm": 25.0,
        "meal": "dinner",
        "hostel": "Boys Hostel 2"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    errors = response.json().get("detail", [])
    assert any("cooked_kg" in str(err) for err in errors)
