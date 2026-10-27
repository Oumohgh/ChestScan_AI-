
import os
import tensorflow as tf
from src.preprocessing import preprocess_image

MODEL_PATH = os.path.join(
    "models",
    "feature_extraction_model.keras"
)

def load_prediction_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

model = load_prediction_model()

def predict_image(image, threshold=0.50):
    processed_img = preprocess_image(image)

    prob = float(
        model.predict(processed_img, verbose=0)[0][0]
    )

    prediction = "PNEUMONIA" if prob >= threshold else "NORMAL"

    return {
        "prediction": prediction,
        "probability": prob,
        "confidence": prob if prediction == "PNEUMONIA" else 1.0 - prob
    }
