import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from fakeredis import FakeAsyncRedis

from src.app.dependencies import verify_api_key
from src.app.main import app



@pytest.fixture
def mock_app_state():
    mock_model = MagicMock()
    mock_model.predict = MagicMock(return_value=None)

    app.state.model_lgbm = mock_model
    app.state.inference_meta = {"features": []}
    app.state.redis = FakeAsyncRedis(decode_responses=True)
    app.dependency_overrides[verify_api_key] = lambda: "fake-test-key"

    yield app

    app.dependency_overrides.clear()


def test_business_rules(mock_app_state):
    client = TestClient(mock_app_state)

    response = client.post(
        "/predict",
        headers={"x-api-key": "123"},
        json={"TransactionAmt": "600000",
              "is_new_device_uid1": "1",
              "TransactionID" : "0",
              "card1": "0"}
    )

    assert response.status_code == 200

    data = response.json()
    assert data["action"] == "BLOCK"
    assert "Blocked by Business Rule: Huge amount from new device" in data["reason"]

