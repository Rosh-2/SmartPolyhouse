import requests
from firebase_admin import credentials, firestore, initialize_app
from datetime import datetime
import base64
import os
import sys

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

# Disease treatment messages
disease_messages = {
    "angular leafspot": """  
- Keep those leaves dry by watering at the base instead of overhead. 
- Gently remove any infected leaves and toss them out. 
- For a natural fix, try spraying neem oil (diluted to 1-2%) or some compost tea—it works wonders!""",
    
    "leaf spot": """
- Go for resistant strawberry varieties to stay ahead of trouble. 
- Clear away affected leaves and keep the area tidy. 
- A quick spray of baking soda mix (1 teaspoon with 1 teaspoon oil in a gallon of water) or garlic extract can help turn things around.""",
    
    "powder mildew": """  
- Give your plants some breathing room with good spacing for air flow. 
- Ease up on nitrogen fertilizer to keep growth in check. 
- A simple milk spray (1 part milk to 9 parts water) or a dusting of sulfur can clear up that powdery mess fast.""",
    
    "strawberry leaf scorch": """  
- Space out your plants to let air move freely and avoid crowding. 
- Water at the base to keep leaves dry and happy. 
- A splash of neem oil or a potassium bicarbonate mix (1 tablespoon per gallon of water) can bring them back to life.""",

    "anthracnose fruit rot": """
- Keep the fruit off the ground with some mulch or straw to stay dry. 
- Pick off any mushy or dark-spotted berries right away. 
- A gentle spray of copper soap or a chamomile tea rinse can help keep this funky rot in check.""",

    "gray mold": """
- Snip off any fuzzy, grayish fruit or leaves as soon as you spot them. 
- Make sure air flows nicely around your plants with good spacing. 
- A quick mist of grapefruit seed extract or a vinegar mix (1 tablespoon per gallon of water) can nudge things back to healthy.""",

    "powdery mildew fruit": """
- Give your berries some space to breathe with proper pruning and airflow. 
- Avoid overwatering and keep the soil just right—not too soggy. 
- A light spray of milk (1 part milk to 9 parts water) or a sprinkle of sulfur powder can chase that powdery coating away."""
}

# Fetch additional info from Firebase
def get_disease_info(disease_name):
    print(f"Fetching additional info for {disease_name} from Firebase...")
    try:
        doc_ref = db.collection("disease_info").document(disease_name).get()
        if doc_ref.exists:
            return doc_ref.to_dict().get("description", "No additional info available.")
        else:
            return "No additional info found in Firebase."
    except Exception as e:
        print(f"❌ Error fetching disease info: {e}")
        return "Error fetching additional info."

# Check if the server is running
def check_server():
    print("Checking server status...")
    try:
        response = requests.get(f'{SERVER_URL}/ping', timeout=5)
        print(f"Ping Status: {response.status_code}, Body: {response.text}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to the server: {e}")
        return False

# Test the /predict route and store in Firestore with Base64 image
def test_prediction(image_path, user_id):
    print(f"Processing image: {image_path} for user: {user_id}")
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
            disease_name = data.get("predicted_disease", "Unknown").replace("_", " ").title()
            confidence = data.get("disease_confidence", 0.0)

            # If the model predicts "Strawberry Healthy", do NOT store it in Firestore
            if disease_name.lower() == "strawberry healthy":
                print("✅ The leaf is healthy. No need to store in Firestore.")
                return

            # Get additional info from Firebase
            firebase_info = get_disease_info(disease_name)

            # Get disease treatment message
            treatment_info = disease_messages.get(disease_name.lower(), "No specific treatment guidelines available.")

            # Combine additional info and treatment info
            additional_info = f"{treatment_info}"

            # Print disease-specific message
            print(f"\n⚠️ Detected Disease: {disease_name}")
            print(treatment_info)
            print(f"\n📌 Additional Info: {additional_info}")

            # Get current date and time
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%I:%M %p")

            # Store in Firestore under user-specific collection
            print("Writing to Firestore...")
            doc_ref = db.collection('users').document(user_id).collection('diseases').add({
                'diseaseName': disease_name,
                'confidence': float(confidence),
                'date': date,
                'time': time,
                'imageBase64': image_base64,
                'processedFrom': 'trial.py',
                'additionalInfo': additional_info
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
    if len(sys.argv) < 2:
        print("❌ Please provide a user_id as a command-line argument.")
        exit(1)
    
    user_id = sys.argv[1]  # Get user_id from command-line argument
    if check_server():
        print("Server check passed, waiting for image path...")
        test_image_path = input("Enter the path of the test image: ").strip()
        test_prediction(test_image_path, user_id)
    else:
        print("Server check failed, exiting.")
