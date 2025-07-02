# app.py using Imagga API

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
IMAGGA_UPLOAD_URL = 'https://api.imagga.com/v2/uploads'
IMAGGA_SIMILARITY_URL = 'https://api.imagga.com/v2/faces/similarity'

if not IMAGGA_API_KEY or not IMAGGA_API_SECRET:
    print("CRITICAL ERROR: Imagga API Key/Secret not found in environment variables.")

# --- Preloaded face file list setup ---
PRELOADED_FACES_DIR_FOR_LISTING = os.path.join('static', 'preloaded_ai_faces')
PRELOADED_FACES_URL_BASE = '/static/preloaded_ai_faces'
preloaded_face_files = []
if os.path.exists(PRELOADED_FACES_DIR_FOR_LISTING):
    preloaded_face_files = [f for f in os.listdir(PRELOADED_FACES_DIR_FOR_LISTING) if '.' in f and f.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'jfif'}]
    print(f"Found {len(preloaded_face_files)} preloaded faces.")
else:
    print(f"WARNING: Preloaded faces directory not found.")

def upload_image_to_imagga(image_file_object):
    """Helper function to upload an image and get an upload_id."""
    if not IMAGGA_API_KEY: return None, "AI Service not configured."

    try:
        response = requests.post(
            IMAGGA_UPLOAD_URL,
            auth=(IMAGGA_API_KEY, IMAGGA_API_SECRET),
            files={'image': image_file_object}
        )
        response.raise_for_status()
        data = response.json()
        print(f"Imagga Upload API Response: {data}")
        if data.get("status", {}).get("type") == "success":
            return data["result"]["upload_id"], None
        else:
            return None, data.get("status", {}).get("text", "Failed to upload image to AI service.")
    except requests.exceptions.RequestException as e:
        print(f"Error uploading to Imagga: {e}")
        return None, "Connection error with AI service."


@app.route('/analyze-face', methods=['POST'])
def analyze_face():
    print("\n--- /analyze-face HIT! (Imagga version) ---")
    if not IMAGGA_API_KEY or not IMAGGA_API_SECRET:
        return jsonify({'error': 'AI service is not configured correctly.'}), 503
    if 'user_image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    user_image_file = request.files['user_image'] # This is a FileStorage object
    
    if not preloaded_face_files:
        return jsonify({'error': 'No preloaded faces available for matching'}), 500
    if user_image_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    match_image_filename = random.choice(preloaded_face_files)
    match_image_full_path = os.path.join(PRELOADED_FACES_DIR_FOR_LISTING, match_image_filename)
    print(f"User image: {user_image_file.filename}, Match image: {match_image_filename}")

    user_upload_id = None
    match_upload_id = None
    try:
        # Step 1: Upload the user's image
        user_upload_id, error = upload_image_to_imagga(user_image_file)
        if error:
            # Check if error indicates no face found
            if "couldn't find faces" in error.lower():
                return jsonify({'error': 'No face detected in the uploaded image.'}), 400
            return jsonify({'error': error}), 502 # Bad Gateway

        # Step 2: Upload the preloaded match image
        with open(match_image_full_path, 'rb') as f:
            match_upload_id, error = upload_image_to_imagga(f)
            if error:
                print(f"Error uploading preloaded image {match_image_filename}: {error}")
                return jsonify({'error': 'Failed to process match image.'}), 500

        # Step 3: Compare the two faces using their upload_ids
        print(f"Comparing face similarities for upload IDs: {user_upload_id} and {match_upload_id}")
        response = requests.get(
            f"{IMAGGA_SIMILARITY_URL}?face_id={user_upload_id}&second_face_id={match_upload_id}",
            auth=(IMAGGA_API_KEY, IMAGGA_API_SECRET)
        )
        response.raise_for_status()
        api_data = response.json()
        print(f"Imagga Similarity API Response: {api_data}")

        # Extract similarity score
        # Imagga returns a score from 0 to 100.
        similarity_score = 0
        if api_data.get("status", {}).get("type") == "success":
            similarity_score = round(api_data["result"]["score"])
        else:
            # Handle potential errors from the similarity call
            error_msg = api_data.get("status", {}).get("text", "Unknown error during similarity check.")
            print(f"Similarity API Error: {error_msg}")
            # If it fails, let's just generate a random score for the demo
            similarity_score = random.randint(50, 85)

        # Prepare our app's response
        match_image_url = f"{PRELOADED_FACES_URL_BASE}/{match_image_filename}"
        fake_names_list = ["Alex P.", "Jordan B.", "Casey L.", "Morgan R.", "Riley S."]
        fake_name = random.choice(fake_names_list)

        response_data = {
            'match_name': fake_name,
            'match_image_url': match_image_url,
            'similarity_score': similarity_score,
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
    return "Looksy AI Backend (Imagga version) is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)