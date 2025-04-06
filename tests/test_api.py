import requests

def test_predict():
    url = "http://localhost:8000/predict/"
    files = {"file": open("static/example.png", "rb")}
    response = requests.post(url, files=files)
    assert response.status_code == 200
    assert "predicted_digit" in response.json()