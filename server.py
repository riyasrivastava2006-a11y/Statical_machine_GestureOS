
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from pathlib import Path
import subprocess
import sys

app = FastAPI(title="GestureOS")

PROJECT_FOLDER = Path(__file__).resolve().parent
WEB_FOLDER = PROJECT_FOLDER / "web"
HTML_FILE = WEB_FOLDER / "index.html"

current_gesture = "None"
current_volume = 50

gesture_process = None
mouse_process = None
volume_process = None


class GestureUpdate(BaseModel):
    gesture: str


class VolumeUpdate(BaseModel):
    volume: int = Field(ge=0, le=100)


def is_running(process):
    return process is not None and process.poll() is None


def stop_process(process):
    if is_running(process):
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def launch_script(filename):
    script_path = PROJECT_FOLDER / filename

    if not script_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Cannot find {filename} in {PROJECT_FOLDER}"
        )

    try:
        return subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_FOLDER)
        )
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not launch {filename}: {error}"
        )


def ensure_camera_is_free(exclude=None):
    processes = {
        "GestureOS": gesture_process,
        "Mouse Control": mouse_process,
        "Volume Control": volume_process
    }

    for name, process in processes.items():
        if name != exclude and is_running(process):
            raise HTTPException(
                status_code=409,
                detail=f"{name} is already running. Stop it first."
            )


@app.get("/")
def home():
    if not HTML_FILE.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"index.html not found at {HTML_FILE}"
        )

    return FileResponse(HTML_FILE)


@app.get("/status")
def status():
    return {
        "gesture": current_gesture,
        "volume": current_volume,
        "system": "GestureOS",
        "status": "Running",
        "gesture_active": is_running(gesture_process),
        "mouse_active": is_running(mouse_process),
        "volume_active": is_running(volume_process)
    }


@app.post("/gesture")
def update_gesture(data: GestureUpdate):
    global current_gesture
    current_gesture = data.gesture
    return {"ok": True, "gesture": current_gesture}


@app.post("/volume/update")
def update_volume(data: VolumeUpdate):
    global current_volume
    current_volume = data.volume
    return {"ok": True, "volume": current_volume}


@app.post("/start")
def start_system():
    global gesture_process, current_gesture

    if is_running(gesture_process):
        return {"message": "GestureOS is already running"}

    ensure_camera_is_free(exclude="GestureOS")
    gesture_process = launch_script("main.py")
    current_gesture = "None"

    return {"message": "GestureOS started"}


@app.post("/stop")
def stop_system():
    global gesture_process, current_gesture

    stop_process(gesture_process)
    gesture_process = None
    current_gesture = "None"

    return {"message": "GestureOS stopped"}


@app.post("/mouse/start")
def start_mouse():
    global mouse_process

    if is_running(mouse_process):
        return {"message": "Mouse Control is already running"}

    ensure_camera_is_free(exclude="Mouse Control")
    mouse_process = launch_script("mouse_control.py")

    return {"message": "Mouse Control started"}


@app.post("/mouse/stop")
def stop_mouse():
    global mouse_process

    stop_process(mouse_process)
    mouse_process = None

    return {"message": "Mouse Control stopped"}


@app.post("/volume/start")
def start_volume():
    global volume_process

    if is_running(volume_process):
        return {"message": "Volume Control is already running"}

    ensure_camera_is_free(exclude="Volume Control")
    volume_process = launch_script("volume_control.py")

    return {"message": "Volume Control started"}


@app.post("/volume/stop")
def stop_volume():
    global volume_process

    stop_process(volume_process)
    volume_process = None

    return {"message": "Volume Control stopped"}


@app.post("/volume/set")
def set_system_volume(data: VolumeUpdate):
    global current_volume

    try:
        result = subprocess.run(
            [
                "osascript",
                "-e",
                f"set volume output volume {data.volume}"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=result.stderr.strip()
                or "Could not change Mac volume."
            )

        current_volume = data.volume

        return {
            "ok": True,
            "volume": current_volume,
            "message": f"System volume set to {current_volume}%."
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="osascript is unavailable."
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Changing system volume timed out."
        )


@app.on_event("shutdown")
def shutdown_cleanup():
    global gesture_process, mouse_process, volume_process

    stop_process(gesture_process)
    stop_process(mouse_process)
    stop_process(volume_process)

    gesture_process = None
    mouse_process = None
    volume_process = None
