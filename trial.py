import requests

# Set the Flask server URL (Change the IP if your server runs on a different machine)
SERVER_URL = "http://34.31.80.242:5000"  # Change to your VM IP if needed

# Check if the server is running
def check_server():
    try:
        response = requests.get(f"{SERVER_URL}/ping")
        if response.status_code == 200:
            print("✅ Server is running!")
        else:
            print(f"⚠ Server responded with status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to the server: {e}")

# Get user input for the image path
test_image_path = input("Enter the path of the test image: ").strip()

# Test the /predict route
def test_prediction(image_path):
    try:
        with open(image_path, "rb") as img_file:
            files = {"file": img_file}
            response = requests.post(f"{SERVER_URL}/predict", files=files)
            print("Response:", response.json())
    except Exception as e:
        print(f"❌ Prediction request failed: {e}")

if __name__ == "__main__":
    check_server()
    test_prediction(test_image_path)
