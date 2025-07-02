# app.py using Microsoft Azure Face API (sending raw octet-stream)

import os
import random
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

# --- Configuration for Azure Face API ---
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_ENDPOINT_URL = os.environ.get("AZURE_ENDPOINT_URL") # This should be the regional endpoint now

if AZURE_ENDPOINT_URL:
    base_url = AZURE_ENDPOINT_URL.rstrip('/')
    AZURE_DETECT_URL = f"{base_url}/face/v1.0/detect"
    AZURE_VERIFY_URL = f"{base_url}/face/v1.0/verify"
else:
    AZURE_DETECT_URL = None
    AZURE_VERIFY_URL = None

if not AZURE_API_KEY or not AZURE_ENDPOINT_URL:
    print("CRITICAL ERROR: Azure API Key/Endpoint URL not found in environment variables.")

# --- Preloaded face file list setup ---
PRELOADED_FACES_DIR_FOR_LISTING = os.path.join('static', 'preloaded_ai_faces')
PRELOADED_FACES_URL_BASE = '/static/preloaded_ai_faces'
preloaded_face_files = []
if os.path.exists(PRELOADED_FACES_DIR_FOR_LISTING):
    preloaded_face_files = [f for f in os.listdir(PRELOADED_FACES_DIR_FOR_LISTING) if '.' in f and f.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'jfif'}]
    print(f"Found {len(preloaded_face_files)} preloaded faces.")
else:
    print(f"WARNING: Preloaded faces directory not found.")


def get_face_id(image_data, api_key, detect_url, filename_for_debug="image"):
    """Helper function to call Azure Detect API by sending raw image data."""
    if not detect_url or not api_key:
        return None, "AI service endpoint or key not configured."
    
    # --- KEY CHANGE: We are now sending raw binary data ---
    # The header must specify 'application/octet-stream' for this method.
    headers = {
        'Ocp-Apim-Subscription-Key': api_key,
        'Content-Type': 'application/octet-stream' 
    }
    # Parameters are sent in the URL, not in a JSON body for this type of request.
    params = {'returnFaceId': 'true', 'returnFaceLandmarks': 'false', 'recognitionModel': 'recognition_04'}
    
    try:
        print(f"--- Calling Azure Detect for {filename_for_debug} using octet-stream ---")
        
        # --- KEY CHANGE: Send the raw bytes in the 'data' parameter ---
        response = requests.post(detect_url, params=params, headers=headers, data=image_data)
        
        response.raise_for_status() # Raise exception for bad status codes
        
        detected_faces = response.json()
        print(f"Azure Detect API response for {filename_for_debug}: {detected_faces}")
        
        if not detected_faces:
            return None, f"No face detected in {filename_for_debug}."
            
        return detected_faces[0]['faceId'], None

    except requests.exceptions.HTTPError as http_err:
        # (Error handling remains the same as before)
        try:
            error_details = http_err.response.json().get('error', {})
            error_message = error_details.get('message', str(http_err))
        except:
            error_message = str(http_err)
        print(f"HTTP Error during face detection for {filename_for_debug}: {error_message}")
        
        if 'Invalid request has been sent' in error_message:
            return None, "Invalid request sent to AI service. Please check image data format."
        if 'InvalidImageSize' in error_message or 'InvalidImageFormat' in error_message:
            return None, "Invalid image format or size. Please use a clear JPG or PNG under 6MB."
        return None, f"AI Service Error: {error_message}"
    except Exception as e:
        print(f"Unexpected error during face detection for {filename_for_debug}: {e}")
        return None, "An unexpected error occurred during face detection."


@app.route('/analyze-face', methods=['POST'])
def analyze_face():
    print("\n--- /analyze-face HIT! (Azure Face API - octet-stream version) ---")
    if not AZURE_API_KEY or not AZURE_VERIFY_URL or not AZURE_DETECT_URL:
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
        # Read the file contents into raw bytes
        user_image_data = user_image_file.read()
        with open(match_image_full_path, 'rb') as f:
            match_image_data = f.read()

        # Step 1: Get faceId for the user's image
        user_face_id, error = get_face_id(user_image_data, AZURE_API_KEY, AZURE_DETECT_URL, filename_for_debug=user_image_file.filename)
        if error:
            return jsonify({'error': error}), 400
        
        # Step 2: Get faceId for the match image
        match_face_id, error = get_face_id(match_image_data, AZURE_API_KEY, AZURE_DETECT_URL, filename_for_debug=match_image_filename)
        if error:
            print(f"Could not detect face in preloaded image {match_image_filename}.")
            return jsonify({'error': 'Error analyzing library image.'}), 500

        # Step 3: Call Verify API to compare the two faceIds (This logic remains the same)
        print(f"Verifying faceId1: {user_face_id} against faceId2: {match_face_id}")
        verify_headers = {'Ocp-Apim-Subscription-Key': AZURE_API_KEY, 'Content-Type': 'application/json'}
        verify_payload = {'faceId1': user_face_id, 'faceId2': match_face_id}
        
        response = requests.post(AZURE_VERIFY_URL, json=verify_payload, headers=verify_headers)
        response.raise_for_status()
        api_data = response.json()
        print(f"Azure Verify API Response: {api_data}")

        similarity_score = round(api_data.get('confidence', 0) * 100)

        # Prepare response (This logic remains the same)
        match_image_url = f"{PRELOADED_FACES_URL_BASE}/{match_image_filename}"
        fake_names_list = ["Alex P.", "Jordan B.", "Casey L.", "Morgan R.", "Riley S."]
        fake_name = random.choice(fake_names_list)

        response_data = { 'match_name': fake_name, 'match_image_url': match_image_url, 'similarity_score': similarity_score, 'match_insta': None }
        return jsonify(response_data), 200

    except requests.exceptions.HTTPError as http_err:
        # (Error handling remains the same)
        try:
            error_details = http_err.response.json().get('error', {})
            error_message = error_details.get('message', str(http_err))
            print(f"HTTP Error from Azure: {error_message}")
            return jsonify({'error': f'AI Service Error: {error_message}'}), 502
        except:
            print(f"HTTP Error calling Azure API: {http_err}")
            return jsonify({'error': 'Could not connect to AI analysis service.'}), 502
            
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/')
def hello_world():
    return "Looksy AI Backend (Azure Face API version) is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)