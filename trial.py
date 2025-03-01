import requests
from firebase_admin import credentials, firestore, initialize_app
from datetime import datetime
import base64
import os

print("Starting script...")

# Initialize Firebase
try:
    print("Initializing Firebase...")
    cred = credentials.Certificate(r"C:\Users\rejir\OneDrive\Desktop\Personal Folder\disease-detector-b18fe-firebase-adminsdk-fbsvc-3c01da0d13.json")
    initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully!")
except Exception as e:
    print(f"❌ Firebase initialization failed: {e}")
    exit(1)

# Set the Flask server URL
SERVER_URL = "http://34.31.80.242:5000"
print(f"Server URL set to: {SERVER_URL}")

# Check if the server is running
def check_server():
    print("Checking server status...")
    try:
        response = requests.get(f"{SERVER_URL}/ping", timeout=5)
        print(f"Ping Status: {response.status_code}, Body: {response.text}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to the server: {e}")
        return False

# Test the /predict route and store in Firestore with Base64 image
def test_prediction(image_path):
    print(f"Processing image: {image_path}")
    try:
        # Convert image to Base64
        print("Converting image to Base64...")
        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        print(f"Image converted to Base64 (length: {len(image_base64)} chars)")

        # Send image to /predict
        print("Opening image file for API...")
        with open(image_path, "rb") as img_file:
            files = {"file": img_file}
            print("Sending request to /predict...")
            response = requests.post(f"{SERVER_URL}/predict", files=files, timeout=10)
            print(f"Response Status: {response.status_code}, Body: {response.text}")
            
            # Extract prediction data
            data = response.json()
            print(f"Parsed JSON: {data}")
            disease_name = data.get("predicted_disease", "Unknown")
            confidence = data.get("confidence", 0.0)

            # Get current date and time
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            # Store in Firestore with Base64 image
            print("Writing to Firestore...")
            doc_ref = db.collection('diseases').add({
                'diseaseName': disease_name,
                'confidence': float(confidence),
                'date': date,
                'time': time,
                'imageBase64': image_base64,  # Store Base64 string
                'processedFrom': 'trial.py'
            })
            print(f"🔥 Successfully stored in Firestore with doc ID: {doc_ref[1].id}")
            
    except FileNotFoundError as e:
        print(f"❌ Image file not found: {e}")
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
    except ValueError as e:
        print(f"❌ JSON parsing failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("Entering main block...")
    if check_server():
        print("Server check passed, waiting for image path...")
        test_image_path = input("Enter the path of the test image: ").strip()
        test_prediction(test_image_path)
    else:
        print("Server check failed, exiting.")
