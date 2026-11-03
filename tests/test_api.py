from fastapi.testclient import TestClient
from backend.main import app
client=TestClient(app)
def test_health():
    r=client.get('/health'); assert r.status_code==200; assert 'model_ready' in r.json()
def test_frontend():
    r=client.get('/'); assert r.status_code==200; assert 'Clasificación' in r.text
def test_invalid_empty_body():
    assert client.post('/predict',json={'body':''}).status_code==422
def test_invalid_missing_body():
    assert client.post('/predict',json={'subject':'Hola'}).status_code==422
def test_prediction_if_model_present():
    from backend.main import MODEL_PATH
    if not MODEL_PATH.is_file():
        import pytest
        pytest.skip('Ejecutar python train_export.py para crear el modelo')
    r=client.post('/predict',json={'subject':'Password reset','body':'I cannot access my account'})
    assert r.status_code==200
    data=r.json()
    assert isinstance(data['queue'],str)
    assert 0<=data['confidence']<=1
    assert isinstance(data['requires_human_review'],bool)
