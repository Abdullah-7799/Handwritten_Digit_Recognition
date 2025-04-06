from pydantic import BaseModel

class PredictionResult(BaseModel):
    predicted_digit: int
    confidence: float
    class_probabilities: list[float]