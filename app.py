# app.py using Face++ API with secure environment variables

import os
import random
import requests # For making API calls
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv # Loads variables from a .env file

# Load environment variables from .env file (for local development)
# This line should be at the top, after imports
load_dotenv() 

# --- Initialize Flask App ---
app = Flask(__name__, static_folder='static')
CORS(app)

# --- Configuration ---
# Securely get credentials from environment variables
# This will read from your .env file locally, and from Render's environment settings when deployed
FACEPLUSPLUS_API_KEY = os.environ.get("FACEPLUSPLUS_API_KEY")
FACEPLUSPLUS_API_SECRET = os.environ.get("FACEPLUSPLUS_API_SECRET")

# Set the correct API endpoint URL for your key's region (e.g., 'api-cn' or 'api-us')
FACEPLUSPLUS_COMPARE_URL = "https://api-cn.faceplusplus.com/compare" 

# Check if the keys were loaded successfully from the environment
if not FACEPLUSPLUS_API_KEY or not FACEPLUSPLUS_API_SECRET:
    print("CRITICAL ERROR: Face++ API Key/Secret not found in environment variables.")
    print("Please ensure you have a .env file with FACEPLUSPLUS_API_KEY and FACEPLUSPLUS_API_SECRET set locally,")
    print("and that they are configured in your hosting service's environment settings.")
    # For a real app, you might want to exit here:
    # import sys
    # sys.exit("Exiting: Missing required API credentials.")

PRELOADED_FACES_DIR_FOR_LISTING = os.path.join('static', 'preloaded_ai_faces')
PRELOADED_FACES_URL_BASE = '/static/preloaded_ai_faces'

# --- Preload face file list ---
preloaded_face_files = []
if os.path.exists(PRELOADED_FACES_DIR_FOR_LISTING):
    # Filter for allowed extensions
    preloaded_face_files = [f for f in os.listdir(PRELOADED_FACES_DIR_FOR_LISTING) if '.' in f and f.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'jfif'}]
    print(f"Found {len(preloaded_face_files)} preloaded faces: {preloaded_face_files}")
else:
    print(f"WARNING: Preloaded faces directory not found at {PRELOADED_FACES_DIR_FOR_LISTING}")


@app.route('/analyze-face', methods=['POST'])
def analyze_face():
    print("\n--- /analyze-face HIT! (Face++ version) ---")
    
    # Re-check for keys on each request in case the app started with them missing
    if not FACEPLUSPLUS_API_KEY or not FACEPLUSPLUS_API_SECRET:
        return jsonify({'error': 'AI service is not configured correctly.'}), 503

    if 'user_image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    user_image_file = request.files['user_image']
    
    if not preloaded_face_files:
        return jsonify({'error': 'No preloaded faces available for matching'}), 500

    if user_image_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Pick a random preloaded face to compare against
    match_image_filename = random.choice(preloaded_face_files)
    match_image_full_path = os.path.join(PRELOADED_FACES_DIR_FOR_LISTING, match_image_filename)
    print(f"User image: {user_image_file.filename}, Match image: {match_image_filename}")

    try:
        # The 'rb' mode means 'read binary', which is correct for image files
        with open(match_image_full_path, 'rb') as match_image_file:
            # Prepare the request payload for Face++
            api_payload = {
                'api_key': FACEPLUSPLUS_API_KEY,
                'api_secret': FACEPLUSPLUS_API_SECRET,
            }
            files_payload = {
                'image_file1': user_image_file,
                'image_file2': match_image_file,
            }
            
            # Make the API call
            response = requests.post(FACEPLUSPLUS_COMPARE_URL, data=api_payload, files=files_payload)
        
        response.raise_for_status() # Raise an exception for bad HTTP status codes (4xx or 5xx)
        api_data = response.json()
        print(f"Face++ API Response: {api_data}")

        # Check for errors returned in the JSON body from Face++
        if 'error_message' in api_data:
            if "NO_FACE_FOUND" in api_data['error_message']:
                error_to_send = "No face detected in the uploaded image. Please try a clear photo."
            else:
                error_to_send = f"AI API Error: {api_data['error_message']}"
            return jsonify({'error': error_to_send}), 400

        # Extract similarity score from the 'confidence' field
        similarity_score = api_data.get('confidence', 0)

        # Prepare our app's response
        match_image_url = f"{PRELOADED_FACES_URL_BASE}/{match_image_filename}"
        fake_names_list = ["Alex P.", "Jordan B.", "Casey L.", "Morgan R.", "Riley S.", "Devon K."]
        fake_name = random.choice(fake_names_list)

        response_data = {
            'match_name': fake_name,
            'match_image_url': match_image_url,
            'similarity_score': round(similarity_score),
            'match_insta': None 
        }
        return jsonify(response_data), 200

    except requests.exceptions.RequestException as e:
        print(f"Error calling Face++ API: {e}")
        return jsonify({'error': 'Could not connect to AI analysis service.'}), 503
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/')
def hello_world():
    return "Looksy AI Backend (Face++ version) is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)