import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

# Define paths
MODEL_PATH = "strawberry_disease_model.h5"
CLASS_INDICES_PATH = "class_indices.npy"

# Load the trained model and class indices
if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_INDICES_PATH):
    raise FileNotFoundError("❌ Model or class indices file not found!")

try:
    model = load_model(MODEL_PATH, compile=False)
    class_indices = np.load(CLASS_INDICES_PATH, allow_pickle=True).item()
    class_names = {v: k for k, v in class_indices.items()}
    print("✅ Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Error loading model: {e}")

# Print versions for debugging
print(f"🔹 TensorFlow Version: {tf.__version__}")
print(f"🔹 Keras Version: {tf.keras.__version__}")

def preprocess_image(image_path):
    """Load and preprocess image for prediction."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and return prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_path = "temp.jpg"
    file.save(image_path)

    try:
        processed_img = preprocess_image(image_path)
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions) * 100)

        response = {
            "predicted_disease": class_names.get(predicted_class, "Unknown"),
            "confidence": confidence
        }
    except Exception as e:
        response = {"error": f"Prediction failed: {str(e)}"}
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)  # Delete temp file

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
