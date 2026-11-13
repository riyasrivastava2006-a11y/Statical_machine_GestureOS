# GestureOS

## Project Overview
GestureOS is a hand-gesture-based system that allows users to control selected Mac functions using computer vision and hand tracking.

## Problem Statement
Traditional computer interaction relies on physical input devices such as a keyboard and mouse. GestureOS explores a touch-free way to interact with a computer using hand gestures.

## Proposed Solution
GestureOS uses MediaPipe to detect hand gestures through a camera and performs corresponding computer actions.

## Features
- Hand gesture recognition
- Launching applications using gestures
- Mouse cursor control using hand movements
- Pinch gesture for clicking
- Volume control using hand position
- Web-based dashboard for controlling the system

## Technologies Used
- Python
- MediaPipe
- OpenCV
- FastAPI
- HTML, CSS and JavaScript

## Gesture Mapping
- Open Palm — Calculator
- Peace Sign — Notes
- One Finger — Safari
- Fist — Finder

## Project Structure
- `server.py` — Backend server
- `main.py` — Gesture recognition
- `mouse_control.py` — Mouse control
- `volume_control.py` — Volume control
- `gesture_state.py` — Gesture state management
- `web/` — Dashboard files
- `hand_landmarker.task` — Hand landmark model

## How to Run

1. Install Python and the required dependencies.
2. Open the project folder in VS Code.
3. Run:

```bash
python3 -m uvicorn server:app --host 127.0.0.1 --port 8000
```

4. Open `http://127.0.0.1:8000` in your browser.
5. Start the system from the dashboard.

## Project Purpose
This project demonstrates the integration of computer vision, gesture recognition and web-based control to create an alternative method of human-computer interaction.
