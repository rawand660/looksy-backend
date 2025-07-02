# app.py using Microsoft Azure Face API (with native http.client)

import os
import random
import http.client, urllib.request, urllib.parse, urllib.error, base64, json # Using native libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

# --- Configuration for Azure Face API ---
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT_URL") # This should be just the hostname, e.g., eastus.api.cognitive.microsoft.com

if not AZURE_API_KEY or not AZURE_ENDPOINT:
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


def get_face_id_native(image_data, api_key, endpoint_host, filename_for_debug="image"):
    """Helper function to call Azure Detect API using Python's native http.client."""
    if not endpoint_host or not api_key:
        return None, "AI service endpoint or key not configured."
    
    headers = {
        'Ocp-Apim-Subscription-Key': api_key,
        'Content-Type': 'application/octet-stream'
    }
    params = urllib.parse.urlencode({
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'recognitionModel': 'recognition_04'
    })
    
    conn = None
    try:
        print(f"--- Calling Azure Detect for {filename_for_debug} using http.client ---")
        print(f"--- Host: {endpoint_host} ---")
        
        conn = http.client.HTTPSConnection(endpoint_host)
        # The path includes the parameters
        path = f"/face/v1.0/detect?{params}"
        conn.request("POST", path, image_data, headers)
        
        response = conn.getresponse()
        print(f"Azure Detect API status: {response.status}, Reason: {response.reason}")
        
        response_data_bytes = response.read()
        response_data_str = response_data_bytes.decode('utf-8')
        print(f"Azure Detect API raw response: {response_data_str}")
        
        response_json = json.loads(response_data_str)

        if response.status < 200 or response.status >= 300:
            # Error case
            error_message = response_json.get('error', {}).get('message', 'Unknown AI service error.')
            print(f"HTTP Error during face detection for {filename_for_debug}: {error_message}")
            return None, f"AI Service Error: {error_message}"
        
        if not response_json:
            return None, f"No face detected in {filename_for_debug}."
            
        return response_json[0]['faceId'], None

    except Exception as e:
        print(f"Unexpected error during native HTTP call for {filename_for_debug}: {e}")
        import traceback; traceback.print_exc()
        return None, "An unexpected error occurred during face detection."
    finally:
        if conn:
            conn.close()

def verify_faces_native(face_id1, face_id2, api_key, endpoint_host):
    """Helper function to call Azure Verify API."""
    headers = {'Ocp-Apim-Subscription-Key': api_key, 'Content-Type': 'application/json'}
    body = json.dumps({'faceId1': face_id1, 'faceId2': face_id2})
    conn = None
    try:
        conn = http.client.HTTPSConnection(endpoint_host)
        conn.request("POST", "/face/v1.0/verify", body, headers)
        response = conn.getresponse()
        response_data_bytes = response.read()
        response_data_str = response_data_bytes.decode('utf-8')
        response_json = json.loads(response_data_str)
        print(f"Azure Verify API Response: {response_json}")
        if response.status < 200 or response.status >= 300:
            raise Exception(response_json.get('error', {}).get('message', 'Verify API failed.'))
        return response_json
    finally:
        if conn:
            conn.close()

@app.route('/analyze-face', methods=['POST'])
def analyze_face():
    print("\n--- /analyze-face HIT! (Azure Face API - native http.client version) ---")
    if not AZURE_API_KEY or not AZURE_ENDPOINT:
        return jsonify({'error': 'AI service is not configured correctly.'}), 503
    if 'user_image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    user_image_file = request.files['user_image']
    
    if not preloaded_face_files or user_image_file.filename == '':
        return jsonify({'error': 'Invalid input file or no preloaded faces.'}), 400

    match_image_filename = random.choice(preloaded_face_files)
    match_image_full_path = os.path.join(PRELOADED_FACES_DIR_FOR_LISTING, match_image_filename)
    print(f"User image: {user_image_file.filename}, Match image: {match_image_filename}")

    try:
        # NOTE: AZURE_ENDPOINT should now be ONLY the hostname, e.g., "eastus.api.cognitive.microsoft.com"
        # It should NOT include "https://"
        endpoint_hostname = AZURE_ENDPOINT.replace("https://", "").rstrip('/')

        user_image_data = user_image_file.read()
        with open(match_image_full_path, 'rb') as f:
            match_image_data = f.read()

        user_face_id, error = get_face_id_native(user_image_data, AZURE_API_KEY, endpoint_hostname, filename_for_debug=user_image_file.filename)
        if error: return jsonify({'error': error}), 400
        
        match_face_id, error = get_face_id_native(match_image_data, AZURE_API_KEY, endpoint_hostname, filename_for_debug=match_image_filename)
        if error: return jsonify({'error': 'Error analyzing library image.'}), 500

        api_data = verify_faces_native(user_face_id, match_face_id, AZURE_API_KEY, endpoint_hostname)
        similarity_score = round(api_data.get('confidence', 0) * 100)

        match_image_url = f"{PRELOADED_FACES_URL_BASE}/{match_image_filename}"
        fake_names_list = ["Alex P.", "Jordan B.", "Casey L.", "Morgan R.", "Riley S."]
        fake_name = random.choice(fake_names_list)

        response_data = {'match_name': fake_name, 'match_image_url': match_image_url, 'similarity_score': similarity_score, 'match_insta': None}
        return jsonify(response_data), 200
            
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/')
def hello_world():
    return "Looksy AI Backend (Azure Face API - native http.client version) is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)