# app.py using Face++ API
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random
import requests # For making API calls

app = Flask(__name__, static_folder='static')
CORS(app)

# --- Configuration ---
# IMPORTANT: Store these securely, e.g., as environment variables on Render
FACEPLUSPLUS_API_KEY = "QPZqehIqlPTcxG26xoUI1jBlMRaw6D4f"
FACEPLUSPLUS_API_SECRET = "qdIvd16PRPk_3ofhEtXeuMG788X9bGvH"
FACEPLUSPLUS_COMPARE_URL = "https://api-cn.faceplusplus.com/compare"

PRELOADED_FACES_DIR_FOR_LISTING = os.path.join('static', 'preloaded_ai_faces')
PRELOADED_FACES_URL_BASE = '/static/preloaded_ai_faces'

# --- Preload face file list (we still need this to pick a match image) ---
preloaded_face_files = []
if os.path.exists(PRELOADED_FACES_DIR_FOR_LISTING):
    preloaded_face_files = os.listdir(PRELOADED_FACES_DIR_FOR_LISTING)
    print(f"Found preloaded faces: {preloaded_face_files}")
else:
    print(f"WARNING: Preloaded faces directory not found.")


@app.route('/analyze-face', methods=['POST'])
def analyze_face():
    print("\n--- /analyze-face HIT! (Face++ version) ---")
    if 'user_image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    user_image_file = request.files['user_image']
    
    if not preloaded_face_files:
        return jsonify({'error': 'No preloaded faces available for matching'}), 500

    # For the demo, we still pick a random preloaded face to compare against
    match_image_filename = random.choice(preloaded_face_files)
    match_image_full_path = os.path.join(PRELOADED_FACES_DIR_FOR_LISTING, match_image_filename)
    print(f"User image: {user_image_file.filename}, Match image: {match_image_filename}")

    try:
        with open(match_image_full_path, 'rb') as match_image_file:
            # Prepare the request to Face++
            response = requests.post(
                FACEPLUSPLUS_COMPARE_URL,
                data={
                    'api_key': FACEPLUSPLUS_API_KEY,
                    'api_secret': FACEPLUSPLUS_API_SECRET,
                },
                files={
                    'image_file1': user_image_file, # The uploaded file object
                    'image_file2': match_image_file, # The preloaded file object
                }
            )
        
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        api_data = response.json()
        print(f"Face++ API Response: {api_data}")

        # Check for errors returned in the JSON body from Face++
        if 'error_message' in api_data:
            # Check for "NO_FACE_FOUND" specifically
            if "NO_FACE_FOUND" in api_data['error_message']:
                error_to_send = "No face detected in the uploaded image. Please try a clear photo."
            else:
                error_to_send = f"AI API Error: {api_data['error_message']}"
            return jsonify({'error': error_to_send}), 400

        # Extract similarity score
        similarity_score = api_data.get('confidence', 0) # 'confidence' is their similarity score from 0-100

        # Prepare our app's response
        match_image_url = f"{PRELOADED_FACES_URL_BASE}/{match_image_filename}"
        fake_names_list = ["Alex P.", "Jordan B.", "Casey L.", "Morgan R."]
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
        return jsonify({'error': 'An internal server error occurred.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)