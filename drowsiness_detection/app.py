from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import os
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import sys

app = Flask(__name__)
app.secret_key = 'your-secure-secret-key-12345'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Initialize dlib
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except Exception as e:
    print(f"Error loading facial landmark predictor: {e}")
    sys.exit(1)

# In-memory user storage (replace with database in production)
users = {}

# Detection thresholds
EYE_AR_THRESHOLD = 0.25
YAWN_THRESHOLD = 0.5

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    return (A + B + C) / (3.0 * D)

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['username'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please enter both username and password', 'danger')
        elif username in users and check_password_hash(users[username]['password'], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not all([username, password, confirm_password]):
            flash('All fields are required', 'danger')
        elif len(username) < 4:
            flash('Username must be at least 4 characters', 'danger')
        elif len(password) < 6:
            flash('Password must be at least 6 characters', 'danger')
        elif password != confirm_password:
            flash('Passwords do not match', 'danger')
        elif username in users:
            flash('Username already exists', 'danger')
        else:
            users[username] = {
                'password': generate_password_hash(password),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route('/detect', methods=['POST'])
def detect():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized', 'message': 'Please login first'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'No image', 'message': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty file', 'message': 'No file selected'}), 400

    filepath = None
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError("Invalid image file")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        
        if len(faces) == 0:
            return jsonify({
                'status': 'NO_FACE',
                'message': 'No face detected in the image'
            })
        
        status = "AWAKE"
        ear, mar = 0, 0
        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            
            left_eye = shape[42:48]
            right_eye = shape[36:42]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            
            mouth = shape[48:68]
            mar = mouth_aspect_ratio(mouth)
            
            if ear < EYE_AR_THRESHOLD or mar > YAWN_THRESHOLD:
                status = "DROWSY"
                break
        
        return jsonify({
            'status': status,
            'ear': round(ear, 4),
            'mar': round(mar, 4),
            'message': f"Status: {status} (EAR: {round(ear, 4)}, MAR: {round(mar, 4)})"
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': f"Error processing image: {str(e)}"
        }), 500
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    try:
        app.run(host='0.0.0.0', port=8000, debug=True)
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)