# app.py using Imagga Visual Similarity API (/compare endpoint)

import os
import random
import requests # For making API calls
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

# --- Configuration for Imagga ---
IMAGGA_API_KEY = os.environ.get("IMAGGA_API_KEY")
IMAGGA_API_SECRET = os.environ.get("IMAGGA_API_SECRET")
# USE THE /compare ENDPOINT
IMAGGA_COMPARE_URL = "https://api.imagga.com/v2/compare"

if not IMAGGA_API_KEY or not IMAGGA_API_SECRET:
    print("CRITICAL ERROR: Imagga API Key/Secret not found in environment variables.")

# --- Preloaded face file list setup (no changes needed here) ---
PRELOADED_FACES_DIR_FOR_LISTING = os.path.join('static', 'preloaded_ai_faces')
PRELOADED_FACES_URL_BASE = '/static/preloaded_ai_faces'
preloaded_face_files = []
if os.path.exists(PRELOADED_FACES_DIR_FOR_LISTING):
    preloaded_face_files = [f for f in os.listdir(PRELOADED_FACES_DIR_FOR_LISTING) if '.' in f and f.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'jfif'}]
    print(f"Found {len(preloaded_face_files)} preloaded faces.")
else:
    print(f"WARNING: Preloaded faces directory not found.")

@app.route('/analyze-face', methods=['POST'])
def analyze_face():
    print("\n--- /analyze-face HIT! (Imagga /compare version) ---")
    if not IMAGGA_API_KEY or not IMAGGA_API_SECRET:
        return jsonify({'error': 'AI service is not configured correctly.'}), 503
    if 'user_image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    user_image_file = request.files['user_image']
    
    if not preloaded_face_files:
        return jsonify({'error': 'No preloaded faces available for matching'}), 500
    if user_image_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    match_image_filename = random.choice(preloaded_face_files)
    match_image_full_path = os.path.join(PRELOADED_FACES_DIR_FOR_LISTING, match_image_filename)
    print(f"User image: {user_image_file.filename}, Match image: {match_image_filename}")

    try:
        # Prepare the multipart/form-data request for /compare
        files_payload = {
            'image1': user_image_file.read(),
            'image2': open(match_image_full_path, 'rb'),
        }
        
        # Make the API call to the /compare endpoint
        response = requests.post(
            IMAGGA_COMPARE_URL,
            files=files_payload,
            auth=(IMAGGA_API_KEY, IMAGGA_API_SECRET)
        )
        
        response.raise_for_status() # Raise exception for 4xx/5xx errors
        api_data = response.json()
        print(f"Imagga Compare API Response: {api_data}")

        # Check for errors returned from Imagga
        if api_data.get('status', {}).get('type') != 'success':
            error_msg = api_data.get('status', {}).get('text', 'Unknown AI API error')
            # The /compare endpoint doesn't have a specific "no face" error, it just compares images.
            # So we rely on a general error check.
            return jsonify({'error': f"AI API Error: {error_msg}"}), 400
        
        # Extract similarity score
        # The /compare endpoint returns a 'percent' score from 0-100
        similarity_score = api_data.get('result', {}).get('percent', 0)

        # Prepare our app's response
        match_image_url = f"{PRELOADED_FACES_URL_BASE}/{match_image_filename}"
        fake_names_list = ["Alex P.", "Jordan B.", "Casey L.", "Morgan R.", "Riley S."]
        fake_name = random.choice(fake_names_list)

        response_data = {
            'match_name': fake_name,
            'match_image_url': match_image_url,
            'similarity_score': round(similarity_score),
            'match_insta': None 
        }
        return jsonify(response_data), 200

    except requests.exceptions.RequestException as e:
        print(f"Error calling Imagga API: {e}")
        return jsonify({'error': 'Could not connect to AI analysis service.'}), 503
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/')
def hello_world():
    return "Looksy AI Backend (Imagga /compare version) is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)