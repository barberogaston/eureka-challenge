from unittest.mock import Mock

from fastapi.testclient import TestClient

from housing_inference.app import app

client = TestClient(app)


def test_model_performance(model_performance):
    response = client.get("/model_performance")
    assert response.status_code == 200
    assert response.json() == model_performance


def test_predict():
    params = {"bar": 3, "baz": 4.9}
    response = client.get("/predict", params=params)
    assert response.status_code == 200


def test_predict_with_extra_features():
    params = {"bar": 3, "baz": 4.9, "qux": 2.0}
    response = client.get("/predict", params=params)
    assert response.status_code == 200


def test_predict_with_missing_features():
    params = {"bar": 3}
    response = client.get("/predict", params=params)
    assert response.status_code == 400


def test_predict_with_unknown_error(mocker):
    model = Mock()
    model.predict = Mock(side_effect=Exception)
    mocker.patch("housing_inference.app.load_model", return_value=model)
    params = {"bar": 3, "baz": 4.9}
    response = client.get("/predict", params=params)
    assert response.status_code == 500
