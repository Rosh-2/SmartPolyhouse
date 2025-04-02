import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

# Define model paths
BINARY_MODEL_PATH = "binary_leaf_fruit_model.h5"
LEAF_MODEL_PATH = "strawberry_disease_model.h5"
FRUIT_MODEL_PATH = "strawberry_disease_model_1.h5"
BINARY_INDICES_PATH = "binary_class_indices.npy"
LEAF_INDICES_PATH = "class_indices.npy"
FRUIT_INDICES_PATH = "class_indices_1.npy"

# Check if all required files exist
required_files = [
    BINARY_MODEL_PATH, LEAF_MODEL_PATH, FRUIT_MODEL_PATH,
    BINARY_INDICES_PATH, LEAF_INDICES_PATH, FRUIT_INDICES_PATH
]

for file_path in required_files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Required file not found: {file_path}")

# Load models and indices
try:
    binary_model = load_model(BINARY_MODEL_PATH, compile=False)
    leaf_model = load_model(LEAF_MODEL_PATH, compile=False)
    fruit_model = load_model(FRUIT_MODEL_PATH, compile=False)
    
    binary_indices = np.load(BINARY_INDICES_PATH, allow_pickle=True).item()
    leaf_indices = np.load(LEAF_INDICES_PATH, allow_pickle=True).item()
    fruit_indices = np.load(FRUIT_INDICES_PATH, allow_pickle=True).item()
    
    binary_names = {v: k for k, v in binary_indices.items()}
    leaf_names = {v: k for k, v in leaf_indices.items()}
    fruit_names = {v: k for k, v in fruit_indices.items()}
        
    print("✅ All models loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Error loading models: {e}")

# Print versions for debugging
print(f"🔹 TensorFlow Version: {tf.__version__}")
print(f"🔹 Keras Version: {tf.keras.__version__}")

def preprocess_image(image_path):
    """Load and preprocess a single image"""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict_type_and_disease(image_path):
    """Two-stage prediction: type (leaf/fruit) then specific disease"""
    # Load and preprocess image
    processed_img = preprocess_image(image_path)
    
    # Stage 1: Predict leaf or fruit
    type_pred = binary_model.predict(processed_img)
    type_class = np.argmax(type_pred)
    type_confidence = float(np.max(type_pred) * 100)
    predicted_type = binary_names[type_class]
    
    # Stage 2: Predict disease based on type
    if predicted_type.lower() == 'leaf':
        predictions = leaf_model.predict(processed_img)
        class_names = leaf_names
        model_type = 'Leaf'
    else:  # fruit
        predictions = fruit_model.predict(processed_img)
        class_names = fruit_names
        model_type = 'Fruit'
    
    disease_class = np.argmax(predictions)
    disease_confidence = float(np.max(predictions) * 100)
    predicted_disease = class_names[disease_class]
    
    return {
        "type": model_type,
        "disease": predicted_disease,
        "type_confidence": type_confidence,
        "disease_confidence": disease_confidence
    }

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and return prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_path = "temp.jpg"
    file.save(image_path)

    try:
        result = predict_type_and_disease(image_path)
        response = {
            "predicted_type": result["type"],
            "predicted_disease": result["disease"],
            "type_confidence": result["type_confidence"],
            "disease_confidence": result["disease_confidence"]
        }
    except Exception as e:
        response = {"error": f"Prediction failed: {str(e)}"}
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)  # Delete temp file

    return jsonify(response)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
