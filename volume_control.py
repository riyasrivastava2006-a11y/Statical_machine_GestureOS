
import cv2
import mediapipe as mp
import subprocess
import time
import threading
import requests
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"
BACKEND = "http://127.0.0.1:8000"
STREAM_PORT = 8001

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


def set_volume(volume):
    volume = max(0, min(100, round(volume / 5) * 5))

    result = subprocess.run(
        ["osascript", "-e", f"set volume output volume {volume}"],
        capture_output=True,
        text=True,
        timeout=3
    )

    if result.returncode != 0:
        print("Could not change system volume:", result.stderr.strip())
        return False

    return True


def report_volume(volume):
    try:
        requests.post(
            f"{BACKEND}/volume/update",
            json={"volume": volume},
            timeout=0.5
        )
    except requests.RequestException:
        pass


def main():
    global latest_jpeg

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    stream_server = CameraHTTPServer(
        ("127.0.0.1", STREAM_PORT),
        CameraStreamHandler
    )
    threading.Thread(
        target=stream_server.serve_forever,
        daemon=True
    ).start()

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1
    )

    camera = None
    current_volume = -1
    last_update_time = 0
    last_timestamp = 0

    print("Volume Control started!")
    print("Move your hand up to increase volume.")
    print("Move your hand down to decrease volume.")

    try:
        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise RuntimeError(
                "Could not open the camera. Check macOS camera permissions."
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
                    data=rgb
                )

                timestamp = max(
                    int(time.monotonic() * 1000),
                    last_timestamp + 1
                )
                last_timestamp = timestamp

                result = detector.detect_for_video(mp_image, timestamp)

                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]

                    for point in hand:
                        x = int(point.x * frame.shape[1])
                        y = int(point.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                    wrist_y = hand[0].y
                    normalized = (0.85 - wrist_y) / (0.85 - 0.15)
                    volume = int(max(0, min(100, normalized * 100)))
                    volume = max(0, min(100, round(volume / 5) * 5))

                    cv2.putText(
                        frame,
                        f"Volume: {volume}%",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                    now = time.monotonic()

                    if now - last_update_time >= 0.25:
                        if volume != current_volume:
                            if set_volume(volume):
                                current_volume = volume
                                report_volume(volume)
                                print(f"Volume: {volume}%")

                        last_update_time = now

                else:
                    cv2.putText(
                        frame,
                        "Show your hand",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                encoded, buffer = cv2.imencode(".jpg", frame)

                if encoded:
                    with frame_lock:
                        latest_jpeg = buffer.tobytes()

    except KeyboardInterrupt:
        print("Volume Control stopped.")

    except Exception as error:
        print(f"Volume control error: {error}")

    finally:
        if camera is not None:
            camera.release()

        with frame_lock:
            latest_jpeg = None

        stop_stream.set()
        stream_server.shutdown()
        stream_server.server_close()
        print("Camera released.")


if __name__ == "__main__":
    main()
