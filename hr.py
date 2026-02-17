# merged_liveness_recognition.py
import cv2
import time
import random
import numpy as np
import os
import face_recognition
from collections import deque

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("Please install mediapipe in your venv: pip install mediapipe\n" + str(e))

# =========================
# Tunable thresholds
# =========================
EYE_AR_THRESH = 0.20        # lower -> stricter blink (0.18~0.24)
BLINKS_REQUIRED = 2
MAR_THRESH = 0.35           # mouth open ratio (0.30~0.45)
YAW_OFFSET = 0.10           # nose X offset from face center (fraction of face width) to count turn
NOD_OFFSET = 0.10           # nose Y offset from face center (fraction of face height) to count down/up
ACTION_TIME_LIMIT = 6.0     # seconds allowed per challenge
COOLDOWN_BETWEEN_ACTIONS = 0.8  # small pause after success/fail

# =========================
# MediaPipe setup
# =========================
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_spec = mp.solutions.drawing_styles

# We’ll use these key indices from FaceMesh
# Eyes (approx pairs for EAR)
LEFT_EYE = (159, 145, 33, 133)   # (upper, lower, outer, inner)
RIGHT_EYE = (386, 374, 362, 263)

# Mouth
LIP_UP = 13
LIP_DOWN = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# Nose tip (for yaw/pitch)
NOSE_TIP = 1

ACTIONS_POOL = ["blink_twice", "open_mouth", "turn_left", "turn_right", "nod"]

# -------------------------
# Face recognition setup (from second script)
# -------------------------
known_encodings = []
known_names = []
known_dir = "data/raw/known"

print("Loading known faces...")
if os.path.isdir(known_dir):
    for filename in os.listdir(known_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(known_dir, filename)
            img = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
else:
    os.makedirs(known_dir, exist_ok=True)
print(f"Loaded {len(known_encodings)} known faces.")

captured_face_image = None
captured_face_encoding = None
typing_name = False
typed_name = ""
highlight_box = None  # (top, right, bottom, left) of face being named

# =========================
# Functions (from first script)
# =========================
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
    x1, x2 = int(max(0, min(xs))), int(min(w-1, max(xs)))
    y1, y2 = int(max(0, min(ys))), int(min(h-1, max(ys)))
    return x1, y1, x2, y2

def draw_fancy_box(img, x1, y1, x2, y2, color, thickness=2, r=12, d=24):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    # top-left
    cv2.line(img, (x1, y1), (x1+d, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1+d), color, thickness)
    # top-right
    cv2.line(img, (x2, y1), (x2-d, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1+d), color, thickness)
    # bottom-left
    cv2.line(img, (x1, y2), (x1+d, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2-d), color, thickness)
    # bottom-right
    cv2.line(img, (x2, y2), (x2-d, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2-d), color, thickness)

def action_text(action):
    return {
        "blink_twice": "Blink twice",
        "open_mouth": "Open your mouth",
        "turn_left": "Turn your head LEFT",
        "turn_right": "Turn your head RIGHT",
        "nod": "Nod UP-DOWN",
    }[action]

def pick_challenges():
    # Always include blink, add 2 random others (no duplicates)
    others = [a for a in ACTIONS_POOL if a != "blink_twice"]
    chosen = ["blink_twice"] + random.sample(others, 2)
    return chosen

# Per-face state (tracked by face id = nearest-nose heuristic)
class FaceState:
    def __init__(self, challenges):
        self.challenges = deque(challenges)     # remaining actions
        self.current_action = self.challenges[0]
        self.action_start = time.time()
        self.success_actions = []
        self.failed = False

        # trackers
        self.blink_count = 0
        self.was_eyes_closed = False

        self.mouth_open_frames = 0
        self.turn_left_frames = 0
        self.turn_right_frames = 0

        self.nod_phase = 0  # 0 waiting down, 1 waiting up
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
            self.challenges.popleft()
        if self.challenges:
            self.current_action = self.challenges[0]
            self.action_start = time.time()
            self.reset_action_trackers()
            return True
        return False  # no more

def match_faces(prev_centers, curr_centers):
    """
    Assign stable IDs based on nearest-center matching.
    Returns mapping: id -> (cx, cy)
    """
    assigned = {}
    used = set()
    # simple greedy
    for fid, (px, py) in prev_centers.items():
        best = None
        best_d = 1e9
        idx = -1
        for i, (cx, cy) in enumerate(curr_centers):
            if i in used: 
                continue
            d = (px - cx)**2 + (py - cy)**2
            if d < best_d:
                best_d = d
                best = (cx, cy)
                idx = i
        if best is not None:
            assigned[fid] = (idx, best)
            used.add(idx)

    # new faces get new ids
    next_id = (max(prev_centers.keys()) + 1) if prev_centers else 1
    for i, (cx, cy) in enumerate(curr_centers):
        if i not in used:
            assigned[next_id] = (i, (cx, cy))
            next_id += 1

    # build new dict of centers by id
    new_centers_by_id = {}
    index_by_id = {}
    for fid, (i, (cx, cy)) in assigned.items():
        new_centers_by_id[fid] = (cx, cy)
        index_by_id[fid] = i
    return index_by_id, new_centers_by_id

# =========================
# Main merged program
# =========================
def main():
    global typing_name, typed_name, captured_face_image, captured_face_encoding, highlight_box
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    # Higher resolution improves landmark stability (tune if laggy)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    face_states = {}          # id -> FaceState
    prev_centers = {}         # id -> (cx, cy)

    # one-time pick (you can re-pick per session)
    session_challenges = pick_challenges()

    with mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,  # better around eyes/lips
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as fm:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)

            # current face centers (nose tip) for id tracking
            curr_centers = []
            face_landmarks = []

            if res.multi_face_landmarks:
                for fl in res.multi_face_landmarks:
                    lm = fl.landmark
                    nx, ny = lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h
                    curr_centers.append((nx, ny))
                    face_landmarks.append(lm)

            # --- Multiple face detection ---
            if res.multi_face_landmarks and len(res.multi_face_landmarks) > 1:
                cv2.putText(frame, "MULTIPLE FACES DETECTED", (50, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                # Optional: instantly fail all faces
                for fs in face_states.values():
                    fs.failed = True

            # assign/track IDs
            index_by_id, prev_centers = match_faces(prev_centers, curr_centers)

            # ensure each id has a state
            # (each new face gets its own copy of the session challenges)
            for fid in index_by_id:
                if fid not in face_states:
                    face_states[fid] = FaceState(challenges=list(session_challenges))

            # draw & evaluate
            for fid, i in index_by_id.items():
                lm = face_landmarks[i]
                x1, y1, x2, y2 = face_bbox(lm, w, h)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                fw, fh = max(1, x2-x1), max(1, y2-y1)

                state = face_states[fid]

                # compute basic measures
                left_ear = eye_aspect_ratio(lm, w, h, LEFT_EYE)
                right_ear = eye_aspect_ratio(lm, w, h, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0
                mar = mouth_aspect_ratio(lm, w, h)

                nose = np.array([lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h])
                box_center = np.array([cx, cy])

                # normalized offsets
                nx_off = (nose[0] - box_center[0]) / (fw + 1e-6)
                ny_off = (nose[1] - box_center[1]) / (fh + 1e-6)

                # --- evaluate current challenge ---
                now = time.time()
                time_left = max(0.0, ACTION_TIME_LIMIT - (now - state.action_start))

                action = state.current_action

                # Blue box during test
                color = (255, 0, 0)
                status_text = f"TESTING: {action_text(action)}  ({time_left:0.1f}s left)"

                # Blink detection (count on open->close transition)
                if action == "blink_twice":
                    if ear < EYE_AR_THRESH:
                        state.was_eyes_closed = True
                    else:
                        if state.was_eyes_closed:
                            state.blink_count += 1
                        state.was_eyes_closed = False

                    if state.blink_count >= BLINKS_REQUIRED:
                        state.success_actions.append(action)
                        if not state.next_action():
                            # all done for this face
                            pass
                        time.sleep(COOLDOWN_BETWEEN_ACTIONS)

                elif action == "open_mouth":
                    if mar > MAR_THRESH:
                        state.mouth_open_frames += 1
                    else:
                        state.mouth_open_frames = 0
                    if state.mouth_open_frames >= 5:
                        state.success_actions.append(action)
                        if not state.next_action():
                            pass
                        time.sleep(COOLDOWN_BETWEEN_ACTIONS)

                elif action == "turn_left":
                    if nx_off < -YAW_OFFSET:
                        state.turn_left_frames += 1
                    else:
                        state.turn_left_frames = 0
                    if state.turn_left_frames >= 5:
                        state.success_actions.append(action)
                        if not state.next_action():
                            pass
                        time.sleep(COOLDOWN_BETWEEN_ACTIONS)

                elif action == "turn_right":
                    if nx_off > YAW_OFFSET:
                        state.turn_right_frames += 1
                    else:
                        state.turn_right_frames = 0
                    if state.turn_right_frames >= 5:
                        state.success_actions.append(action)
                        if not state.next_action():
                            pass
                        time.sleep(COOLDOWN_BETWEEN_ACTIONS)

                elif action == "nod":
                    # need down then up relative to center
                    if state.nod_phase == 0:
                        if ny_off > NOD_OFFSET:
                            state.nod_down_ok = True
                            state.nod_phase = 1
                    elif state.nod_phase == 1:
                        if ny_off < -NOD_OFFSET:
                            state.nod_up_ok = True

                    if state.nod_down_ok and state.nod_up_ok:
                        state.success_actions.append(action)
                        if not state.next_action():
                            pass
                        time.sleep(COOLDOWN_BETWEEN_ACTIONS)

                # timeout check
                if (time.time() - state.action_start) > ACTION_TIME_LIMIT and state.current_action == action:
                    state.failed = True

                # decide label color
                if not state.challenges:  # all succeeded => LIVE
                    color = (0, 200, 0)
                    label = "LIVE ✔"

                    # ===== Face recognition stage (from second script), ONLY when live =====
                    # Crop the live face region
                    x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w-1, x2), min(h-1, y2)
                    live_face_img = frame[y1c:y2c, x1c:x2c]
                    name = "Unknown"

                    if live_face_img.size != 0:
                        # Convert ROI to RGB and encode
                        rgb_face = cv2.cvtColor(live_face_img, cv2.COLOR_BGR2RGB)
                        face_encs = face_recognition.face_encodings(rgb_face)
                        if face_encs:
                            face_encoding = face_encs[0]
                            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                            if True in matches:
                                name = known_names[matches.index(True)]

                            # Draw recognition-style overlays (keep second script UI)
                            if name == "Unknown":
                                color_rec = (0, 0, 255)  # red
                                cv2.putText(frame, "Press 'C' to capture", (x1c, max(0, y1c - 10)),
                                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)
                            else:
                                color_rec = (0, 255, 0)  # green

                            # Rectangle + name bar (like 2nd script)
                            cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), color_rec, 2)
                            cv2.rectangle(frame, (x1c, y2c - 35), (x2c, y2c), color_rec, cv2.FILLED)
                            cv2.putText(frame, name, (x1c + 6, y2c - 6),
                                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

                            # If we press C on unknown face (keep original behavior: check key here)
                            if (not typing_name) and (name == "Unknown"):
                                # NOTE: keep the inline waitKey check exactly like the second script
                                if (cv2.waitKey(1) & 0xFF) == ord('c'):
                                    captured_face_image = live_face_img.copy()
                                    captured_face_encoding = face_encoding
                                    typing_name = True
                                    typed_name = ""
                                    # highlight_box expects (top, right, bottom, left)
                                    highlight_box = (y1c, x2c, y2c, x1c)

                elif state.failed:
                    color = (0, 0, 255)
                    label = "SPOOF / FAILED ✖"
                else:
                    label = status_text

                # Fancy box + labels from first script
                draw_fancy_box(frame, x1, y1, x2, y2, color, 2)

                # action progress dots/text from first script
                done = len(state.success_actions)
                total = len(session_challenges)
                cv2.putText(frame, f"{label}", (x1, max(20, y1-18)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA)
                cv2.putText(frame, f"Progress: {done}/{total}", (x1, y2+20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

            # Draw typing overlay (from second script)
            if typing_name:
                overlay = frame.copy()
                cv2.rectangle(overlay, (50, 50), (600, 150), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cv2.putText(frame, "Enter name: " + typed_name, (60, 120),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

                # Highlight the face we're naming (from second script)
                if highlight_box:
                    ht, hr, hb, hl = highlight_box
                    cv2.rectangle(frame, (hl, ht), (hr, hb), (255, 0, 0), 4)

            # Top banner (from first script)
            cv2.putText(frame, "Challenge-Response Liveness | Press 'r' to reshuffle challenges, 'q' to quit",
                        (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            cv2.imshow("Liveness (Challenge-Response)", frame)

            # Global key processing (keep both scripts' keys)
            key = cv2.waitKey(1) & 0xFF

            # Typing flow (from second script)
            if typing_name:
                if key == 13:  # Enter
                    if typed_name.strip() and captured_face_image is not None and captured_face_encoding is not None:
                        save_path = os.path.join(known_dir, f"{typed_name.strip()}.jpg")
                        cv2.imwrite(save_path, captured_face_image)
                        known_encodings.append(captured_face_encoding)
                        known_names.append(typed_name.strip())
                        print(f"[INFO] Saved {typed_name.strip()} to database.")
                    typing_name = False
                    highlight_box = None
                    captured_face_image = None
                    captured_face_encoding = None
                elif key == 8:  # Backspace
                    typed_name = typed_name[:-1]
                elif key != 255 and key != 0:
                    try:
                        typed_name += chr(key)
                    except:
                        pass  # ignore non-printable

            # Session reshuffle (from first script)
            if key == ord('r'):
                session = pick_challenges()
                for fs in face_states.values():
                    fs.__init__(list(session))

            # Quit (both scripts)
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
 