from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from threading import Thread
from queue import Queue
import cv2
import numpy as np
import base64
from pathlib import Path

from pose.pose import PoseGestureController

app = FastAPI()
frame_queue: Queue[np.ndarray] = Queue()
controller = PoseGestureController()

html_path = Path(__file__).with_name("client.html")

@app.get("/")
async def index():
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>No client.html found</h1>")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            if data.startswith("data:image"):
                _, b64data = data.split(",", 1)
                img_bytes = base64.b64decode(b64data)
                npimg = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame_queue.put(frame)
            await ws.send_text("ok")
    except WebSocketDisconnect:
        pass


def _frame_generator():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        yield frame


def start(host: str = "0.0.0.0", port: int = 8000):
    def run_controller():
        controller.start(frame_generator=_frame_generator())

    controller_thread = Thread(target=run_controller, daemon=True)
    controller_thread.start()

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start()
