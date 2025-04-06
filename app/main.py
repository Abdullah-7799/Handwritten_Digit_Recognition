from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os

# Disable oneDNN warnings if needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = FastAPI(
    title="Handwritten Digit Recognition API",
    description="API for recognizing handwritten digits using CNN model",
    version="1.0.0"
)

# Load model function
def load_model():
    """Load the pre-trained CNN model"""
    try:
        model = tf.keras.models.load_model('app/mnist_cnn.h5')
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

model = load_model()

@app.get("/")
async def root():
    return {"message": "Digit Recognition API - Send POST request to /predict with an image file"}

@app.post("/predict/")
async def predict_digit(file: UploadFile = File(...)):
    """
    Recognize handwritten digit from image
    
    Parameters:
    - file: UploadFile - Image file (JPEG/PNG) containing handwritten digit
    
    Returns:
    - JSON response with prediction results
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload JPEG or PNG image.")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('L')
        image = image.resize((28, 28))
        
        # Prepare image array
        img_array = np.array(image)
        img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        return {
            "predicted_digit": predicted_class,
            "confidence": confidence,
            "class_probabilities": prediction[0].tolist()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# For development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)