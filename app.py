from flask import Flask, Request, render_template, jsonify, Response
import cv2
import time
import random
import numpy as np
import os
import face_recognition
from collections import deque
import mediapipe as mp
import base64
from datetime import datetime
import threading

# Global variables and configuration
app = Flask(__name__, static_folder='static', template_folder='templates')

camera = None
known_encodings = []
known_names = []
known_dir = "data/raw/known"

# =========================
# Liveness and Recognition Configuration
# =========================
EYE_AR_THRESH = 0.20
BLINKS_REQUIRED = 2
MAR_THRESH = 0.35
YAW_OFFSET = 0.10
NOD_OFFSET = 0.10
ACTION_TIME_LIMIT = 6.0
COOLDOWN_BETWEEN_ACTIONS = 0.8
ACTIONS_POOL = ["blink_twice", "open_mouth", "turn_left", "turn_right", "nod"]

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_spec = mp.solutions.drawing_styles
LEFT_EYE = (159, 145, 33, 133)
RIGHT_EYE = (386, 374, 362, 263)
LIP_UP = 13
LIP_DOWN = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
NOSE_TIP = 1

# =========================
# Helper Functions
# =========================
def pick_challenges():
    """
    Selects a set of liveness challenges, always including 'blink_twice'.
    """
    others = [a for a in ACTIONS_POOL if a != "blink_twice"]
    chosen = random.sample(others, 2)
    return ["blink_twice"] + chosen

def eye_aspect_ratio(landmarks, w, h, idxs):
    up, down, outer, inner = idxs
    p_up = np.array([landmarks[up].x * w, landmarks[up].y * h])
    p_down = np.array([landmarks[down].x * w, landmarks[down].y * h])
    p_outer = np.array([landmarks[outer].x * w, landmarks[outer].y * h])
    p_inner = np.array([landmarks[inner].x * w, landmarks[inner].y * h])
    vert = np.linalg.norm(p_up - p_down)
    horiz = np.linalg.norm(p_outer - p_inner) + 1e-6
    return vert / horiz

def mouth_aspect_ratio(landmarks, w, h):
    up = np.array([landmarks[LIP_UP].x * w, landmarks[LIP_UP].y * h])
    down = np.array([landmarks[LIP_DOWN].x * w, landmarks[LIP_DOWN].y * h])
    left = np.array([landmarks[MOUTH_LEFT].x * w, landmarks[MOUTH_LEFT].y * h])
    right = np.array([landmarks[MOUTH_RIGHT].x * w, landmarks[MOUTH_RIGHT].y * h])
    vert = np.linalg.norm(up - down)
    horiz = np.linalg.norm(left - right) + 1e-6
    return vert / horiz

def face_bbox(landmarks, w, h):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, x2 = int(max(0, min(xs))), int(min(w - 1, max(xs)))
    y1, y2 = int(max(0, min(ys))), int(min(h - 1, max(ys)))
    return x1, y1, x2, y2

def action_text(action):
    return {
        "blink_twice": "Blink twice",
        "open_mouth": "Open your mouth",
        "turn_left": "Turn your head LEFT and RIGHT",
        "turn_right": "Turn your head RIGHT and LEFT",
        "nod": "Nod UP-DOWN",
    }[action]

def load_known_faces():
    global known_encodings, known_names
    print("Loading known faces...")
    if not os.path.isdir(known_dir):
        os.makedirs(known_dir, exist_ok=True)
        print(f"Created known faces directory: {known_dir}")
        return

    known_encodings = []
    known_names = []
    for filename in os.listdir(known_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(known_dir, filename)
            img = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
    print(f"Loaded {len(known_encodings)} known faces.")

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open webcam.")
            return None
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera

# =========================
# Liveness State and Threading
# =========================
class LivenessState:
    def __init__(self):
        self.challenges = deque(pick_challenges())
        self.current_action = self.challenges[0] if self.challenges else None
        self.action_start = time.time()
        self.success_actions = []
        self.failed = False
        self.result = "pending"
        self.result_label = "Ready to start liveness check"

        self.blink_count = 0
        self.was_eyes_closed = False
        self.mouth_open_frames = 0
        self.turn_left_frames = 0
        self.turn_right_frames = 0
        self.nod_phase = 0
        self.nod_down_ok = False
        self.nod_up_ok = False

    def reset_action_trackers(self):
        self.blink_count = 0
        self.was_eyes_closed = False
        self.mouth_open_frames = 0
        self.turn_left_frames = 0
        self.turn_right_frames = 0
        self.nod_phase = 0
        self.nod_down_ok = False
        self.nod_up_ok = False

    def next_action(self):
        if self.challenges:
            self.success_actions.append(self.challenges.popleft())
        if self.challenges:
            self.current_action = self.challenges[0]
            self.action_start = time.time()
            self.reset_action_trackers()
            return True
        return False

liveness_state = LivenessState()
liveness_lock = threading.Lock()



# =========================
# Liveness State and Threading
# =========================


def liveness_detection_thread():
    global liveness_state
    
    # Add counters for stabilization
    consecutive_multiple_faces_frames = 0
    consecutive_no_faces_frames = 0
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as fm:
        while True:
            camera_instance = get_camera()
            if camera_instance is None:
                time.sleep(1)
                continue

            ok, frame = camera_instance.read()
            if not ok:
                time.sleep(1)
                continue
            
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)

            with liveness_lock:
                if liveness_state.result != "pending":
                    # Reset counters when not in pending state
                    consecutive_no_faces_frames = 0
                    consecutive_multiple_faces_frames = 0
                    time.sleep(1)
                    continue

                # =======================================================
                # RE-INTEGRATED CODE: Use a stabilization buffer (recommended)
                # =======================================================
                if not res.multi_face_landmarks:
                    consecutive_no_faces_frames += 1
                    if consecutive_no_faces_frames > 5: # Fails after 5 consecutive frames
                        liveness_state.failed = True
                        liveness_state.result = "failed"
                        liveness_state.result_label = "No face detected."
                    continue # Continue to the next frame to keep counting
                else:
                    consecutive_no_faces_frames = 0
                
                if len(res.multi_face_landmarks) > 1:
                    consecutive_multiple_faces_frames += 1
                    if consecutive_multiple_faces_frames > 5: # Fails after 5 consecutive frames
                        liveness_state.failed = True
                        liveness_state.result = "failed"
                        liveness_state.result_label = "Multiple faces detected."
                    continue # Continue to the next frame to keep counting
                else:
                    consecutive_multiple_faces_frames = 0
                # =======================================================

                lm = res.multi_face_landmarks[0].landmark
                
                ear = (eye_aspect_ratio(lm, w, h, LEFT_EYE) + eye_aspect_ratio(lm, w, h, RIGHT_EYE)) / 2.0
                mar = mouth_aspect_ratio(lm, w, h)
                
                x1, y1, x2, y2 = face_bbox(lm, w, h)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                fw, fh = max(1, x2-x1), max(1, y2-y1)
                nx_off = (lm[NOSE_TIP].x * w - cx) / (fw + 1e-6)
                ny_off = (lm[NOSE_TIP].y * h - cy) / (fh + 1e-6)
                
                state = liveness_state
                action = state.current_action

                # Evaluate the current challenge
                if action == "blink_twice":
                    if ear < EYE_AR_THRESH:
                        state.was_eyes_closed = True
                    elif state.was_eyes_closed:
                        state.blink_count += 1
                        state.was_eyes_closed = False
                    if state.blink_count >= BLINKS_REQUIRED:
                        state.next_action()
                        time.sleep(COOLDOWN_BETWEEN_ACTIONS)

                elif action == "open_mouth":
                    if mar > MAR_THRESH:
                        state.mouth_open_frames += 1
                    else:
                        state.mouth_open_frames = 0
                    if state.mouth_open_frames >= 5:
                        state.next_action()
                        time.sleep(COOLDOWN_BETWEEN_ACTIONS)
                        
                elif action == "turn_left":
                    if nx_off < -YAW_OFFSET:
                        state.turn_left_frames += 1
                    else:
                        state.turn_left_frames = 0
                    if state.turn_left_frames >= 5:
                        state.next_action()
                        time.sleep(COOLDOWN_BETWEEN_ACTIONS)

                elif action == "turn_right":
                    if nx_off > YAW_OFFSET:
                        state.turn_right_frames += 1
                    else:
                        state.turn_right_frames = 0
                    if state.turn_right_frames >= 5:
                        state.next_action()
                        time.sleep(COOLDOWN_BETWEEN_ACTIONS)

                elif action == "nod":
                    if state.nod_phase == 0 and ny_off > NOD_OFFSET:
                        state.nod_down_ok = True
                        state.nod_phase = 1
                    elif state.nod_phase == 1 and ny_off < -NOD_OFFSET:
                        state.nod_up_ok = True
                    if state.nod_down_ok and state.nod_up_ok:
                        state.next_action()
                        time.sleep(COOLDOWN_BETWEEN_ACTIONS)

                # Timeout check
                if (time.time() - state.action_start) > ACTION_TIME_LIMIT and state.result == "pending":
                    state.failed = True
                    state.result = "failed"
                    state.result_label = "Spoof Detected."

                if not state.challenges and not state.failed:
                    state.result = "live"
                    state.result_label = "Liveness Check Passed"





# =========================
# Flask Routes
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        camera_instance = get_camera()
        if not camera_instance:
            return
        while True:
            success, frame = camera_instance.read()
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_liveness_status')
def get_liveness_status():
    with liveness_lock:
        state = liveness_state
        progress = len(state.success_actions)
        total = len(liveness_state.challenges) + progress
        if total == 0: total = len(ACTIONS_POOL)
        
        return jsonify({
            "status": state.result,
            "label": state.result_label,
            "current_action": action_text(state.current_action) if state.current_action else "N/A",
            "progress": f"{progress}/{total}",
            "time_left": max(0.0, ACTION_TIME_LIMIT - (time.time() - state.action_start))
        })
    

from flask import Flask, render_template, jsonify, Response, request


# =========================
# New Flask Route for Registration
# =========================
@app.route('/register_face', methods=['POST'])
def register_face():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data received. Check Content-Type header."}), 400
            
        name = data.get('name')
        photo_data_url = data.get('photo')
        
        if not name or not photo_data_url:
            return jsonify({"status": "error", "message": "Name and photo data are required."}), 400

        # Split the data URL to get the Base64 part
        header, encoded_data = photo_data_url.split(',')
        
        # Decode the Base64 image string
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error", "message": "Failed to decode image data."}), 400

        # Get the face encoding from the captured frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)
        
        if not face_encodings:
            return jsonify({"status": "error", "message": "Could not find a face in the provided photo."}), 400

        face_encoding = face_encodings[0]

        # Save the new person's data
        with liveness_lock:
            known_encodings.append(face_encoding)
            known_names.append(name.strip())
            
        # Also save the image file to the known directory
        save_path = os.path.join(known_dir, f"{name.strip()}.jpg")
        cv2.imwrite(save_path, frame)
        
        return jsonify({
            "status": "success",
            "message": f"Successfully registered face and fingerprint for {name.strip()}.",
            "name": name.strip(),
            "photo": photo_data_url
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"An internal server error occurred: {str(e)}"}), 500

# ... (all your existing Flask routes and the __main__ block)

@app.route('/capture_and_process', methods=['POST'])
def capture_and_process():
    with liveness_lock:
        status = liveness_state.result
        
    if status != "live":
        return jsonify({
            "status": "pending_liveness",
            "result_label": "Please complete the liveness check first.",
            "is_unique": False,
            "name": "N/A",
            "photo": ""
        })

    camera_instance = get_camera()
    if not camera_instance:
        return jsonify({"status": "error", "message": "Camera not available."}), 500

    ok, frame = camera_instance.read()
    if not ok:
        return jsonify({"status": "error", "message": "Could not read from camera."}), 500

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_encodings = face_recognition.face_encodings(rgb)
    
    if not face_encodings:
         return jsonify({
            "status": "no_face",
            "result_label": "No face detected.",
            "is_unique": False,
            "name": "N/A",
            "photo": ""
        })
    
    if len(face_encodings) > 1:
        return jsonify({
            "status": "multiple_faces",
            "result_label": "Multiple faces detected.",
            "is_unique": False,
            "name": "N/A",
            "photo": ""
        })

    face_encoding = face_encodings[0]
    name = "Unknown"
    is_unique = True

    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)

    if True in matches:
        first_match_index = matches.index(True)
        name = known_names[first_match_index]
        is_unique = False

    _, buffer = cv2.imencode('.jpg', frame)
    photo_base64 = base64.b64encode(buffer).decode('utf-8')
    photo_data_url = f"data:image/jpeg;base64,{photo_base64}"
    
    # Reset liveness state for the next session
    with liveness_lock:
        liveness_state.__init__()

    if not is_unique:
        return jsonify({
            "status": "known",
            "result_label": f"IDENTIFIED: {name}",
            "is_unique": False,
            "name": name,
            "photo": photo_data_url
        })
    else:
        return jsonify({
            "status": "unknown",
            "result_label": "UNIQUE FACE",
            "is_unique": True,
            "name": "Unknown",
            "photo": photo_data_url
        })

@app.route('/start_liveness_check', methods=['POST'])
def start_liveness_check():
    with liveness_lock:
        liveness_state.__init__()
    return jsonify({"message": "Liveness check started."})
    
if __name__ == '__main__':
    load_known_faces()
    liveness_thread = threading.Thread(target=liveness_detection_thread, daemon=True)
    liveness_thread.start()
    app.run(debug=True)