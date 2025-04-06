```markdown
# Handwritten Digit Recognition API with CNN and FastAPI

![Demo](static/example.png)

A production-ready API for recognizing handwritten digits (0-9) using a Convolutional Neural Network (CNN) and FastAPI framework.

## Features

- 🎯 **Accurate predictions** with CNN model (98%+ accuracy on MNIST)
- ⚡ **Fast inference** with optimized TensorFlow backend
- 📚 **Auto-generated documentation** (Swagger UI & ReDoc)
- 🐳 **Docker support** for easy deployment
- ✅ **Input validation** and error handling
- 📊 **Probability distribution** for predictions

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/handwritten-digit-recognition.git
cd handwritten-digit-recognition

# Install dependencies
pip install -r requirements.txt

# Train model (if not using pre-trained)
python app/model.py
```

### Running the API
```bash
uvicorn app.main:app --reload
```
Access interactive docs at: http://localhost:8000/docs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/predict/` | POST | Predict digit from image |

### Example Request
```bash
curl -X POST -F "file=@static/example.png" http://localhost:8000/predict/
```

### Example Response
```json
{
  "predicted_digit": 7,
  "confidence": 0.9934,
  "class_probabilities": [0.0001, ..., 0.9934, ...]
}
```

## Project Structure

```
handwritten-digit-recognition/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── model.py         # CNN model training
│   └── model.py         # For pydantic models
├── static/              # Sample images
│   └── example.png
├── tests/               # Test cases
│   └── test_api.py
├── requirements.txt     # Dependencies
├── Dockerfile           # Containerization
└── README.md
```

## Deployment

### Docker
```bash
docker build -t digit-recognition .
docker run -p 8000:8000 digit-recognition
```

### Production (with Gunicorn)
```bash
gunicorn -k uvicorn.workers.UvicornWorker app.main:app
```

## Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments
- MNIST dataset
- TensorFlow/Keras
- FastAPI team
