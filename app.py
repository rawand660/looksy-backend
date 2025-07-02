# app.py using face_recognition with on-demand caching

import os
import random
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

try:
    import face_recognition
    import numpy as np
    from PIL import Image
    print("AI libraries imported successfully.")
    AI_AVAILABLE = True
except Exception as e:
    print(f"!!!!!!!!!! CRITICAL ERROR IMPORTING AI LIBRARIES: {e} !!!!!!!!!!!")
    AI_AVAILABLE = False

# --- Configuration ---
PRELOADED_FACES_DIR = os.path.join('static', 'preloaded_ai_faces')
PRELOADED_FACES_URL_BASE = '/static/preloaded_ai_faces'

# --- In-Memory Cache (will be populated on the first request) ---
preloaded_face_encodings_cache = {} # { 'filename.jpg': encoding_vector, ... }
is_cache_loaded = False # A flag to check if we've loaded the cache yet

def load_and_cache_all_encodings_if_needed():
    """Function to populate the cache if it hasn't been loaded yet."""
    global preloaded_face_encodings_cache, is_cache_loaded
    if is_cache_loaded or not AI_AVAILABLE:
        return

    print("--- First request: Loading and caching all preloaded face encodings... ---")
    if not os.path.exists(PRELOADED_FACES_DIR):
        print(f"WARNING: Preloaded faces directory not found: {PRELOADED_FACES_DIR}")
        return

    valid_image_files = [f for f in os.listdir(PRELOADED_FACES_DIR) if '.' in f and f.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'jfif'}]
    
    for filename in valid_image_files:
        try:
            img_path = os.path.join(PRELOADED_FACES_DIR, filename)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                preloaded_face_encodings_cache[filename] = encodings[0]
                print(f"Cached encoding for {filename}.")
            else:
                print(f"Could not find a face in preloaded file: {filename}")
        except Exception as e:
            print(f"Error caching encoding for {filename}: {e}")

    is_cache_loaded = True # Set the flag so we don't do this again
    print(f"Finished caching. {len(preloaded_face_encodings_cache)} encodings cached.")


@app.route('/analyze-face', methods=['POST'])
def analyze_face():
    print("\n--- /analyze-face HIT! ---")
    
    # --- KEY CHANGE: Load the cache on the first request ---
    if not is_cache_loaded:
        load_and_cache_all_encodings_if_needed()

    if not AI_AVAILABLE:
        return jsonify({'error': 'AI processing service is currently unavailable.'}), 503
    if not preloaded_face_encodings_cache:
        return jsonify({'error': 'AI model data not ready or no preloaded faces available.'}), 503

    if 'user_image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    user_image_file = request.files['user_image']
    if user_image_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        print(f"Processing user image: {user_image_file.filename}")
        user_image = face_recognition.load_image_file(user_image_file)
        user_encodings = face_recognition.face_encodings(user_image)

        if not user_encodings:
            return jsonify({'error': 'No face detected in the uploaded image. Please try a clear photo of a face.'}), 400
        
        user_encoding = user_encodings[0]
        print("User encoding generated.")

        # Compare against the now-populated cache
        known_encodings = list(preloaded_face_encodings_cache.values())
        filenames = list(preloaded_face_encodings_cache.keys())
        
        face_distances = face_recognition.face_distance(known_encodings, user_encoding)

        matches = []
        for i, distance in enumerate(face_distances):
            similarity_score = int(max(0, (1 - (distance / 0.75))) * 100)
            similarity_score = min(max(similarity_score, 40), 99)
            matches.append({'filename': filenames[i], 'distance': distance, 'similarity_score': similarity_score})
        
        sorted_matches = sorted(matches, key=lambda x: x['distance'])
        print(f"Found {len(sorted_matches)} potential matches, sorted by distance.")

        if not sorted_matches:
            return jsonify({'error': 'No matches found after comparison.'}), 500

        top_n_matches = 5
        response_matches = []
        fake_names_list = ["Alex P.", "Jordan B.", "Casey L.", "Morgan R.", "Riley S.", "Devon K."]
        for i, match_data in enumerate(sorted_matches[:top_n_matches]):
            fake_name = fake_names_list[i % len(fake_names_list)] 
            response_matches.append({
                'match_name': f"{fake_name} (Match #{i+1})",
                'match_image_url': f"{PRELOADED_FACES_URL_BASE}/{match_data['filename']}",
                'similarity_score': match_data['similarity_score'],
                'distance': round(match_data['distance'], 4),
                'match_insta': None
            })

        return jsonify({'matches': response_matches}), 200

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred during analysis.'}), 500

@app.route('/')
def hello_world():
    return "Looksy AI Backend (face_recognition/lazy-load) is running!"

# We no longer call load_and_cache... at the global level.
# The `if __name__ == '__main__':` block is only for local testing.
# Gunicorn will directly use the 'app' object.
if __name__ == '__main__':
    print("--- Starting Flask App for LOCAL development ---")
    app.run(host='0.0.0.0', port=5001, debug=True)