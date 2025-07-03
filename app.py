# app.py using face_recognition with corrected similarity calculation

import os
import random
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

AI_AVAILABLE = False
try:
    import face_recognition
    import numpy as np
    from PIL import Image
    print("AI libraries (face_recognition, numpy) imported successfully.")
    AI_AVAILABLE = True
except Exception as e:
    print(f"!!!!!!!!!! CRITICAL ERROR IMPORTING AI LIBRARIES: {e} !!!!!!!!!!!")

preloaded_face_encodings_cache = {}
is_cache_loaded = False

PRELOADED_FACES_DIR = os.path.join('static', 'preloaded_ai_faces')
PRELOADED_FACES_URL_BASE = '/static/preloaded_ai_faces'

def load_and_cache_all_encodings_if_needed():
    global preloaded_face_encodings_cache, is_cache_loaded
    if is_cache_loaded or not AI_AVAILABLE:
        return

    print("--- First request: Loading and caching all preloaded face encodings... ---")
    if not os.path.exists(PRELOADED_FACES_DIR):
        print(f"WARNING: Preloaded faces directory not found: {PRELOADED_FACES_DIR}")
        is_cache_loaded = True
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

    is_cache_loaded = True
    print(f"Finished caching. {len(preloaded_face_encodings_cache)} encodings cached.")

@app.route('/analyze-face', methods=['POST'])
def analyze_face():
    print("\n--- /analyze-face HIT! ---")
    
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

        known_encodings = list(preloaded_face_encodings_cache.values())
        filenames = list(preloaded_face_encodings_cache.keys())
        face_distances = face_recognition.face_distance(known_encodings, user_encoding)

        matches = []
        for i, distance in enumerate(face_distances):
            # --- CORRECTED SIMILARITY CALCULATION ---
            # Linear mapping: distance 0.0 -> ~99%, distance 0.6 -> ~63%, distance 1.0 -> ~39%
            similarity_score = 99 - (distance * 60)
            
            # Clamp the score to our desired demo range of 40-99
            similarity_score = int(min(max(similarity_score, 40), 99))
            
            matches.append({'filename': filenames[i], 'distance': distance, 'similarity_score': similarity_score})
        
        sorted_matches = sorted(matches, key=lambda x: x['distance']) # Sort by distance (lower is better)
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

if __name__ == '__main__':
    print("--- Starting Flask App for LOCAL development ---")
    app.run(host='0.0.0.0', port=5001, debug=True)