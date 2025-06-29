print("--- DEEPFACE AI BACKEND app.py SCRIPT START (Sorted Similarity) ---")

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random
import uuid
import time
import hashlib

try:
    from deepface import DeepFace
    from scipy.spatial.distance import cosine
    import numpy as np
    print("DeepFace and Scipy imported successfully.")
    DEEPFACE_AVAILABLE = True
except ImportError as e:
    print(f"!!!!!!!!!! ERROR IMPORTING LIBRARIES !!!!!!!!!!! Error: {e}")
    DEEPFACE_AVAILABLE = False
except Exception as e_other_import:
    print(f"!!!!!!!!!! UNEXPECTED ERROR DURING IMPORTS !!!!!!!!!!! Error: {e_other_import}")
    DEEPFACE_AVAILABLE = False

app = Flask(__name__, static_folder='static')
print("Flask app object created.")
CORS(app)
print("CORS initialized.")

UPLOAD_FOLDER = 'uploads'
PRELOADED_FACES_DIR_FOR_LISTING = os.path.join('static', 'preloaded_ai_faces')
PRELOADED_FACES_URL_BASE = '/static/preloaded_ai_faces'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jfif'}
MODEL_NAME = 'VGG-Face'
USER_IMAGE_DETECTOR = 'retinaface'

print(f"UPLOAD_FOLDER: {UPLOAD_FOLDER}")
print(f"PRELOADED_FACES_DIR: {PRELOADED_FACES_DIR_FOR_LISTING}")
print(f"Using DeepFace model: {MODEL_NAME}, User image detector: {USER_IMAGE_DETECTOR}")
print(f"DeepFace available status: {DEEPFACE_AVAILABLE}")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PRELOADED_FACES_DIR_FOR_LISTING, exist_ok=True)
print("Directories ensured.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_hash(file_storage):
    hash_sha256 = hashlib.sha256()
    original_position = file_storage.tell()
    file_storage.seek(0)
    for chunk in iter(lambda: file_storage.read(4096), b""):
        hash_sha256.update(chunk)
    file_storage.seek(original_position)
    return hash_sha256.hexdigest()
print("Helper functions defined.")

# --- Cache for preloaded face embeddings ---
preloaded_face_embeddings_cache = {} # Store as { 'filename.jpg': embedding_vector, ... }
preloaded_face_files_list = [] # Just the list of filenames

def load_and_cache_preloaded_embeddings():
    global preloaded_face_files_list, preloaded_face_embeddings_cache
    preloaded_face_embeddings_cache = {} # Clear cache before reloading
    preloaded_face_files_list = []

    if not DEEPFACE_AVAILABLE:
        print("DeepFace not available, cannot load preloaded embeddings.")
        return

    if not os.path.exists(PRELOADED_FACES_DIR_FOR_LISTING):
        print(f"WARNING: Preloaded faces directory not found: {PRELOADED_FACES_DIR_FOR_LISTING}")
        return

    print(f"Loading and caching embeddings for preloaded faces in: {os.path.abspath(PRELOADED_FACES_DIR_FOR_LISTING)}")
    all_files = os.listdir(PRELOADED_FACES_DIR_FOR_LISTING)
    valid_image_files = [f for f in all_files if allowed_file(f)]
    preloaded_face_files_list = valid_image_files # Store the list of valid filenames
    
    if not valid_image_files:
        print("No valid image files found in preloaded directory to cache.")
        return

    for filename in valid_image_files:
        try:
            img_path = os.path.join(PRELOADED_FACES_DIR_FOR_LISTING, filename)
            print(f"Processing preloaded file for cache: {filename}...")
            embedding_obj = DeepFace.represent(img_path=img_path, model_name=MODEL_NAME, enforce_detection=False) # Assuming these are good faces
            if embedding_obj and isinstance(embedding_obj, list) and len(embedding_obj) > 0 and embedding_obj[0].get('embedding'):
                preloaded_face_embeddings_cache[filename] = embedding_obj[0]['embedding']
                print(f"Cached embedding for {filename}.")
            else:
                print(f"Could not generate/cache embedding for preloaded file: {filename}")
        except Exception as e:
            print(f"Error caching embedding for {filename}: {e}")
    print(f"Finished caching preloaded embeddings. {len(preloaded_face_embeddings_cache)} embeddings cached.")

if DEEPFACE_AVAILABLE:
    try:
        print(f"Warming up {MODEL_NAME} model (minimal)...") # Warm-up with a generic call
        # This initial call helps load TensorFlow and model structures.
        # Actual embeddings for preloaded faces will be cached by load_and_cache_preloaded_embeddings.
        DeepFace.build_model(MODEL_NAME) 
        print(f"{MODEL_NAME} model structure built (warm-up).")
        load_and_cache_preloaded_embeddings() # Load and cache at startup
    except Exception as e:
        print(f"ERROR DURING DEEPFACE WARM-UP/PRELOAD CACHING: {e}")
else:
    print("DeepFace not available, skipping model warm-up and preloading.")


@app.route('/analyze-face', methods=['POST'])
def analyze_face():
    request_time_start = time.time()
    print(f"\n--- [{time.strftime('%Y-%m-%d %H:%M:%S')}] /analyze-face HIT! ---")
    user_image_path = None

    if not DEEPFACE_AVAILABLE:
        return jsonify({'error': 'AI processing service unavailable.'}), 503
    if not preloaded_face_embeddings_cache: # Check if cache is populated
        return jsonify({'error': 'AI model data not ready or no preloaded faces available.'}), 503


    try:
        if 'user_image' not in request.files:
            return jsonify({'error': 'No "user_image" in request'}), 400
        file = request.files['user_image']
        file_hash = get_file_hash(file) # Call this before file.save()
        print(f"Received file: {file.filename}, Hash: {file_hash}")

        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file or type'}), 400
            
        extension = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"user_{file_hash[:8]}_{uuid.uuid4().hex[:4]}.{extension}"
        user_image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(user_image_path) # Save after get_file_hash reset the pointer
        print(f"User image saved: {user_image_path}")

        print(f"Getting embedding for user image (detector: {USER_IMAGE_DETECTOR}, enforce=True)...")
        user_embedding_obj = DeepFace.represent(img_path=user_image_path, model_name=MODEL_NAME, detector_backend=USER_IMAGE_DETECTOR, enforce_detection=True)
        if not user_embedding_obj or not user_embedding_obj[0].get('embedding'):
            raise ValueError("Could not get embedding for user image.")
        user_embedding = user_embedding_obj[0]['embedding']
        print("User embedding generated.")

        # Calculate similarity against all preloaded faces
        matches = []
        for filename, preloaded_embedding in preloaded_face_embeddings_cache.items():
            distance = cosine(np.array(user_embedding), np.array(preloaded_embedding))
            
            # Deterministic Similarity Score Mapping
            if MODEL_NAME == 'VGG-Face':
                if distance <= 0.1:   similarity_score = 99 - int(distance * 50) 
                elif distance <= 0.25:  similarity_score = 94 - int((distance - 0.1) * (10/0.15))
                elif distance <= 0.4:  similarity_score = 84 - int((distance - 0.25) * (10/0.15))
                elif distance <= 0.55:  similarity_score = 74 - int((distance - 0.4) * (9/0.15)) 
                elif distance <= 0.7:  similarity_score = 65 - int((distance - 0.55) * (9/0.15)) 
                elif distance <= 0.85:  similarity_score = 56 - int((distance - 0.7) * (8/0.15)) 
                else:                  similarity_score = 48 - int(max(0, (distance - 0.85)) * (8/0.15))
            else: 
                similarity_score = int(max(0, (1.0 - distance / 1.2)) * 60 + 40)
            similarity_score = min(max(similarity_score, 40), 99)

            matches.append({
                'filename': filename,
                'distance': distance,
                'similarity_score': similarity_score
            })
        
        # Sort matches by similarity score (descending) or distance (ascending)
        sorted_matches = sorted(matches, key=lambda x: x['similarity_score'], reverse=True)
        # To sort by distance (closer is better):
        # sorted_matches = sorted(matches, key=lambda x: x['distance'])
        
        print(f"Found {len(sorted_matches)} potential matches, sorted by similarity.")

        if not sorted_matches:
            # This should not happen if preloaded_face_embeddings_cache is not empty
            return jsonify({'error': 'No matches found after comparison.'}), 500

        # For this demo, we will return a list of top N matches to the frontend
        # The frontend can then decide how to display them or cycle
        top_n_matches = 5 # Or how many you want to send
        response_matches = []
        fake_names_list = ["Alex P.", "Jordan B.", "Casey L.", "Morgan R.", "Riley S.", "Devon K.", "Sam W.", "Taylor M.", "Chris J.", "Jamie T."] # More names

        for i, match_data in enumerate(sorted_matches[:top_n_matches]):
            # Ensure enough fake names or cycle through them
            fake_name = fake_names_list[i % len(fake_names_list)] 
            response_matches.append({
                'match_name': f"{fake_name} (Match #{i+1})", # Add rank for clarity
                'match_image_url': f"{PRELOADED_FACES_URL_BASE}/{match_data['filename']}",
                'similarity_score': match_data['similarity_score'],
                'distance': round(match_data['distance'], 4), # Include distance for debugging
                'match_insta': None # Placeholder
            })

        response_data = {'matches': response_matches} # Send a list of matches
        request_time_end = time.time()
        print(f"Returning top {len(response_matches)} matches. (Route time: {request_time_end - request_time_start:.2f}s)")
        return jsonify(response_data), 200

    except ValueError as ve:
        print(f"VALUE ERROR in /analyze-face: {str(ve)}")
        if "face could not be detected" in str(ve).lower() or "model input" in str(ve).lower():
            error_msg_to_send = 'No face detected or image unsuitable. Try a clear photo of a face.'
        else:
            error_msg_to_send = 'Error analyzing image features.'
        return jsonify({'error': error_msg_to_send}), 400

    except Exception as e:
        print(f"GENERAL EXCEPTION in /analyze-face:")
        import traceback; traceback.print_exc()
        return jsonify({'error': 'Internal server error during analysis.'}), 500
    finally:
        if user_image_path and os.path.exists(user_image_path):
            try: os.remove(user_image_path); print(f"Cleaned up: {user_image_path}")
            except Exception as e_clean: print(f"Error cleaning up {user_image_path}: {e_clean}")

@app.route('/')
def hello_world():
    return "Looksy DeepFace AI Backend (Sorted Matches) is running!"

print("Routes defined.")
if __name__ == '__main__':
    print("--- Starting Flask App (Sorted Matches) ---")
    # Important: Flask dev server reloader might cause load_and_cache_preloaded_embeddings
    # to run twice. For production, this caching would be done once.
    # For dev, ensure use_reloader=False if caching at startup causes issues with it running multiple times
    # or if DeepFace/TensorFlow has issues with the reloader's separate process.
    # However, for code changes to app.py, reloader is convenient.
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=True)