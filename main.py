
import cv2
import mediapipe as mp
import time
import subprocess
import requests
import threading
from pathlib import Path
from collections import deque, Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_FOLDER / "hand_landmarker.task"

gesture_history = deque(maxlen=5)

gesture_actions = {
    "OPEN PALM": "Calculator",
    "PEACE SIGN": "Notes",
    "ONE FINGER": "Safari",
    "FIST": "Finder",
}

last_action_time = 0
last_triggered_gesture = None
last_sent_gesture = None
cooldown = 2

latest_jpeg = None
frame_lock = threading.Lock()
stop_stream = threading.Event()


class CameraStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/video_feed":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            while not stop_stream.is_set():
                with frame_lock:
                    frame_data = latest_jpeg

                if frame_data is None:
                    time.sleep(0.03)
                    continue

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    f"Content-Length: {len(frame_data)}\r\n\r\n".encode()
                )
                self.wfile.write(frame_data)
                self.wfile.write(b"\r\n")
                time.sleep(0.03)

        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, format, *args):
        pass


class CameraHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def open_app(app_name):
    try:
        subprocess.Popen(["open", "-a", app_name])
        print(f"Opening {app_name}!")
    except Exception as error:
        print(f"Could not open {app_name}: {error}")


def recognize_gesture(landmarks):
    thumb = abs(landmarks[4].x - landmarks[3].x) > 0.04

    index = landmarks[8].y < landmarks[6].y
    middle = landmarks[12].y < landmarks[10].y
    ring = landmarks[16].y < landmarks[14].y
    pinky = landmarks[20].y < landmarks[18].y

    fingers = [thumb, index, middle, ring, pinky]
    count = sum(fingers)

    if count == 5:
        return "OPEN PALM"
    elif count == 0:
        return "FIST"
    elif index and middle and not ring and not pinky:
        return "PEACE SIGN"
    elif index and not middle and not ring and not pinky:
        return "ONE FINGER"

    return "UNKNOWN"


def main():
    global latest_jpeg
    global last_action_time, last_triggered_gesture, last_sent_gesture

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    stream_server = CameraHTTPServer(
        ("127.0.0.1", 8001),
        CameraStreamHandler
    )
    stream_thread = threading.Thread(
        target=stream_server.serve_forever,
        daemon=True
    )
    stream_thread.start()

    camera = None

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
    )

    try:
        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise RuntimeError(
                "Camera could not be opened. Check macOS camera permissions."
            )

        with HandLandmarker.create_from_options(options) as detector:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read camera frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp = int(time.monotonic() * 1000)
                result = detector.detect_for_video(mp_image, timestamp)

                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]

                    for point in hand:
                        x = int(point.x * frame.shape[1])
                        y = int(point.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                    raw_gesture = recognize_gesture(hand)
                    gesture_history.append(raw_gesture)

                    stable_gesture, frequency = Counter(
                        gesture_history
                    ).most_common(1)[0]

                    if stable_gesture != last_sent_gesture:
                        try:
                            requests.post(
                                "http://127.0.0.1:8000/gesture",
                                json={"gesture": stable_gesture},
                                timeout=0.5,
                            )
                            last_sent_gesture = stable_gesture
                        except requests.RequestException:
                            pass

                    cv2.putText(
                        frame,
                        stable_gesture,
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    now = time.monotonic()

                    if (
                        frequency >= 4
                        and stable_gesture in gesture_actions
                        and stable_gesture != last_triggered_gesture
                        and now - last_action_time >= cooldown
                    ):
                        app_name = gesture_actions[stable_gesture]
                        open_app(app_name)
                        last_action_time = now
                        last_triggered_gesture = stable_gesture

                        cv2.putText(
                            frame,
                            "ACTION: " + app_name,
                            (20, 85),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 0),
                            2,
                        )
                else:
                    gesture_history.clear()

                    if last_sent_gesture != "None":
                        try:
                            requests.post(
                                "http://127.0.0.1:8000/gesture",
                                json={"gesture": "None"},
                                timeout=0.5,
                            )
                            last_sent_gesture = "None"
                        except requests.RequestException:
                            pass

                    last_triggered_gesture = None

                encoded, buffer = cv2.imencode(".jpg", frame)

                if encoded:
                    with frame_lock:
                        latest_jpeg = buffer.tobytes()

    except KeyboardInterrupt:
        print("GestureOS stopped.")

    finally:
        if camera is not None:
            camera.release()

        with frame_lock:
            latest_jpeg = None

        stop_stream.set()
        stream_server.shutdown()
        stream_server.server_close()
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()
