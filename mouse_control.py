import cv2
import mediapipe as mp
import pyautogui
import time
import math
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------- CONFIGURATION ----------------

MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

STREAM_HOST = "127.0.0.1"
STREAM_PORT = 8001

screen_width, screen_height = pyautogui.size()

# Camera stream variables
latest_jpeg = None
frame_lock = threading.Lock()
stop_stream = threading.Event()
stop_stream.clear()

# Mouse settings
SMOOTHING = 0.30
PINCH_THRESHOLD = 0.32
RELEASE_THRESHOLD = 0.45

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ---------------- CAMERA STREAM SERVER ----------------

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
                    frame = latest_jpeg

                if frame is None:
                    time.sleep(0.03)
                    continue

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    f"Content-Length: {len(frame)}\r\n\r\n".encode()
                )
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")

                time.sleep(0.03)

        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, format, *args):
        pass


class CameraHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# ---------------- MAIN MOUSE CONTROL ----------------

def main():
    global latest_jpeg

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Hand tracking model not found: {MODEL_PATH}"
        )

    stream_server = CameraHTTPServer(
        (STREAM_HOST, STREAM_PORT),
        CameraStreamHandler
    )

    stream_thread = threading.Thread(
        target=stream_server.serve_forever,
        daemon=True
    )
    stream_thread.start()

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=1
    )

    camera = None
    last_timestamp = 0
    pinch_active = False
    smooth_x = None
    smooth_y = None

    print("GestureOS Mouse Control Started!")
    print("Move your index finger to move the cursor.")
    print("Pinch your thumb and index finger to click.")
    print("Camera preview is available on the dashboard.")

    try:
        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise RuntimeError(
                "Could not open the camera. Check macOS camera permissions."
            )

        with HandLandmarker.create_from_options(options) as detector:

            while not stop_stream.is_set():

                success, frame = camera.read()

                if not success:
                    print("Could not read camera frame.")
                    break

                # Mirror the camera image
                frame = cv2.flip(frame, 1)

                height, width, _ = frame.shape

                # Convert frame for MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = max(
                    int(time.monotonic() * 1000),
                    last_timestamp + 1
                )
                last_timestamp = timestamp

                result = detector.detect_for_video(
                    mp_image,
                    timestamp
                )

                if result.hand_landmarks:

                    hand = result.hand_landmarks[0]

                    # Finger coordinates
                    index_x = hand[8].x
                    index_y = hand[8].y

                    thumb_x = hand[4].x
                    thumb_y = hand[4].y

                    # Draw hand landmarks
                    for point in hand:
                        px = int(point.x * width)
                        py = int(point.y * height)

                        cv2.circle(
                            frame,
                            (px, py),
                            4,
                            (0, 255, 0),
                            -1
                        )

                    # Calculate cursor target position
                    target_x = int(
                        max(0.08, min(0.92, index_x)) * screen_width
                    )
                    target_y = int(
                        max(0.08, min(0.92, index_y)) * screen_height
                    )

                    # Smooth cursor movement
                    if smooth_x is None:
                        smooth_x = target_x
                        smooth_y = target_y
                    else:
                        smooth_x += SMOOTHING * (target_x - smooth_x)
                        smooth_y += SMOOTHING * (target_y - smooth_y)

                    # Move cursor
                    pyautogui.moveTo(
                        int(smooth_x),
                        int(smooth_y)
                    )

                    # Detect pinch gesture
                    pinch_distance = math.hypot(
                        index_x - thumb_x,
                        index_y - thumb_y
                    )

                    palm_size = math.hypot(
                        hand[9].x - hand[0].x,
                        hand[9].y - hand[0].y
                    )

                    if palm_size > 0:

                        pinch_ratio = pinch_distance / palm_size

                        if (
                            pinch_ratio < PINCH_THRESHOLD
                            and not pinch_active
                        ):
                            pyautogui.click()
                            pinch_active = True
                            print("Pinch detected — click!")

                        elif pinch_ratio > RELEASE_THRESHOLD:
                            pinch_active = False

                    # Highlight index fingertip
                    cv2.circle(
                        frame,
                        (int(index_x * width), int(index_y * height)),
                        10,
                        (255, 0, 255),
                        -1
                    )

                    cv2.putText(
                        frame,
                        "Mouse Control Active",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Pinch to Click",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                else:
                    pinch_active = False

                    cv2.putText(
                        frame,
                        "Show your hand",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                # Encode frame for dashboard preview
                encoded, buffer = cv2.imencode(".jpg", frame)

                if encoded:
                    with frame_lock:
                        latest_jpeg = buffer.tobytes()

    except KeyboardInterrupt:
        print("Mouse control stopped.")

    except Exception as error:
        print(f"Mouse control error: {error}")
        print(
            "If cursor movement or clicking is blocked, "
            "check macOS Accessibility permissions."
        )

    finally:
        if camera is not None:
            camera.release()

        with frame_lock:
            latest_jpeg = None

        stop_stream.set()
        stream_server.shutdown()
        stream_server.server_close()

        print("Camera released.")
        print("GestureOS Mouse Control Closed.")


if __name__ == "__main__":
    main()