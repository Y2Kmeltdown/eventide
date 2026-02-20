#!/usr/bin/python3

import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "eglfs")

import sys
import argparse
import logging
import time

from multiprocessing import Process, Lock, shared_memory

from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow,
                             QPushButton, QHBoxLayout, QVBoxLayout, QWidget)
from PyQt5.QtCore import Qt, QMetaObject, Q_ARG, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

import numpy as np
import neuromorphic_drivers as nd

logging.basicConfig()
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("server")
logger.setLevel(getattr(logging, log_level))

data_lock = Lock()

app = QApplication(sys.argv)
app.setOverrideCursor(Qt.BlankCursor)  # hides cursor app-wide

screen_geo = app.primaryScreen().geometry()
W = screen_geo.width()
H = screen_geo.height()
print(f"DEBUG screen size: {W}x{H}", flush=True)

CONTROLS_W = 200
PREVIEW_W = W - CONTROLS_W
PREVIEW_H = H

# Poll the shared memory at 50fps
POLL_INTERVAL_MS = 20


# ---------------------------------------------------------------------------
# Event camera helpers
# ---------------------------------------------------------------------------

def eventProducer(serial, config, dims, event_shared_memory):
    frame = np.zeros((dims[1], dims[0]), dtype=np.uint8) + 127
    oldTime = time.monotonic_ns()
    with nd.open(serial=serial, configuration=config) as device:
        print(f"Successfully started EVK4 {serial}", flush=True)
        for status, packet in device:
            if packet and packet.polarity_events is not None:
                if packet.polarity_events.size != 0:
                    frame[
                        packet.polarity_events["y"],
                        packet.polarity_events["x"],
                    ] = packet.polarity_events["on"] * 255

                    if time.monotonic_ns() - oldTime >= (1 / 50) * 1_000_000_000:
                        with data_lock:
                            event_shared_memory.buf[:] = frame.tobytes()
                        frame = np.zeros((dims[1], dims[0]), dtype=np.uint8) + 127
                        oldTime = time.monotonic_ns()


def check_event_camera(serialNumberList):
    evkSerialList = [i.serial for i in nd.list_devices()]
    try:
        return [i for i in evkSerialList if i in serialNumberList]
    except Exception as e:
        print(f"Error during serial number check: {e}", flush=True)
        return None


def get_frame(eventMemory: shared_memory.SharedMemory, height: int, width: int) -> np.ndarray:
    """Read the latest event frame from shared memory as a (H, W) uint8 numpy array."""
    with data_lock:
        raw = bytes(eventMemory.buf[:width * height])
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width))


# ---------------------------------------------------------------------------
# Qt widgets
# ---------------------------------------------------------------------------

class FrameDisplay(QLabel):
    """
    Displays numpy arrays (H,W), (H,W,3) RGB, or (H,W,4) RGBA.
    Call update_frame() from any thread safely.
    """

    def __init__(self, width, height, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: black;")
        self._w = width
        self._h = height

    @pyqtSlot(object)
    def _set_pixmap_safe(self, pixmap):
        self.setPixmap(pixmap)

    def update_frame(self, frame: np.ndarray):
        """Thread-safe frame update. Accepts (H,W), (H,W,3), or (H,W,4) uint8 arrays."""
        pixmap = self._to_pixmap(frame)
        if pixmap is None:
            return
        pixmap = pixmap.scaled(self._w, self._h,
                               Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)
        QMetaObject.invokeMethod(self, "_set_pixmap_safe",
                                 Qt.QueuedConnection,
                                 Q_ARG(object, pixmap))

    def _to_pixmap(self, frame: np.ndarray):
        if not isinstance(frame, np.ndarray):
            print(f"update_frame: unsupported type {type(frame)}")
            return None

        if frame.ndim == 2:
            # Grayscale → replicate to RGB
            frame = np.stack([frame] * 3, axis=-1)

        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        h, w, ch = frame.shape

        if ch == 3:
            fmt = QImage.Format_RGB888
        elif ch == 4:
            fmt = QImage.Format_RGBA8888
        else:
            print(f"Unsupported channel count: {ch}")
            return None

        qimg = QImage(frame.data, w, h, w * ch, fmt)
        return QPixmap.fromImage(qimg)


class MainWindow(QMainWindow):
    def __init__(self, shm: shared_memory.SharedMemory, cam_height: int, cam_width: int):
        super().__init__()
        self._shm = shm
        self._cam_h = cam_height
        self._cam_w = cam_width

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setGeometry(0, 0, W, H)

        central = QWidget()
        self.setCentralWidget(central)

        h_layout = QHBoxLayout(central)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)

        # --- Frame display (left) ---
        self.display = FrameDisplay(PREVIEW_W, PREVIEW_H)
        h_layout.addWidget(self.display)

        # --- Controls panel (right) ---
        controls = QWidget()
        controls.setFixedWidth(CONTROLS_W)
        controls.setStyleSheet("background: black;")
        v_layout = QVBoxLayout(controls)
        v_layout.setContentsMargins(8, 16, 8, 16)
        v_layout.setSpacing(12)

        self.label = QLabel("Waiting for frames...")
        self.label.setStyleSheet("font-size: 14px; color: white;")
        self.label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label.setWordWrap(True)
        v_layout.addWidget(self.label)

        v_layout.addStretch()

        # Placeholder button
        self.button = QPushButton("Action")
        self.button.setStyleSheet(
            "font-size: 20px; padding: 12px; background: #444; color: white; border-radius: 6px;"
        )
        self.button.clicked.connect(self.on_button_clicked)
        v_layout.addWidget(self.button)

        h_layout.addWidget(controls)

        # Poll shared memory at POLL_INTERVAL_MS
        self._timer = QTimer()
        self._timer.setInterval(POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll_frame)
        self._timer.start()

        self._frame_count = 0
        self._last_fps_time = time.monotonic()

        self.show()

    def _poll_frame(self):
        """Called by QTimer on the main thread — reads shared memory and updates display."""
        frame = get_frame(self._shm, self._cam_h, self._cam_w)
        self.display.update_frame(frame)

        # Update FPS counter in label roughly every second
        self._frame_count += 1
        now = time.monotonic()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            fps = self._frame_count / elapsed
            self.label.setText(f"FPS: {fps:.1f}\n{self._cam_w}x{self._cam_h}")
            self._frame_count = 0
            self._last_fps_time = now

    def on_button_clicked(self):
        # Placeholder — add functionality here
        self.label.setText("Button pressed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--serial",
        default="",
        help="Event camera serial number(s) (e.g. 00050423 00051505)",
        nargs="+",
        type=str,
    )
    args = parser.parse_args()

    configuration = nd.prophesee_evk4.Configuration(
        biases=nd.prophesee_evk4.Biases(
            diff_off=102,
            diff_on=73,
        ),
        rate_limiter=nd.prophesee_evk4.RateLimiter(
            reference_period_us=200,
            maximum_events_per_period=6000,
        ),
    )

    if args.serial == "":
        evkSerialList = [i.serial for i in nd.list_devices()]
    else:
        evkSerialList = check_event_camera(args.serial)

    if not evkSerialList:
        print("[INFO] No Event Cameras connected to system.", flush=True)
        sys.exit(1)

    # Get sensor dimensions before starting the process
    with nd.open(serial=evkSerialList[0]) as device:
        cam_width = device.properties().width
        cam_height = device.properties().height

    print(f"Event camera: {cam_width}x{cam_height}", flush=True)

    shm_event_data = shared_memory.SharedMemory(create=True, size=cam_width * cam_height)
    eventProcess = Process(
        target=eventProducer,
        args=(evkSerialList[0], configuration, (cam_width, cam_height), shm_event_data),
        daemon=True,
    )

    window = MainWindow(shm_event_data, cam_height, cam_width)

    try:
        eventProcess.start()
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        logger.warning("Keyboard Interrupt, exiting...")
    finally:
        eventProcess.join()
        shm_event_data.close()
        shm_event_data.unlink()