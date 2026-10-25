#!/usr/bin/env python3
"""
Jetson PPE gate with Arduino feedback.

Vision -> 3 s temporal validation -> command (GO / STOP / UNKNOWN) -> Arduino
Arduino -> ACK:<cmd>, GATE:OPEN/CLOSED (heartbeat every 500 ms), EVENT:TIMEOUT

The Jetson uses the feedback to decide:
  - resend a command the Arduino did not acknowledge
  - GO sent but gate never opened  -> "GATE DID NOT OPEN", send STOP, retry later
  - STOP sent but gate stays open  -> "GATE DID NOT CLOSE" alarm, keep sending STOP
  - no messages for 2 s            -> "ARDUINO NOT RESPONDING"
  - shows ACCESS GRANTED only when the gate is confirmed open

Usage:
  python ppe_uart_feedback.py --model best.pt --show-vests
  python ppe_uart_feedback.py --model best.pt --uart-port /dev/ttyUSB0 --instant-stop
"""

import argparse
import logging
import threading
import time
import warnings

import cv2
import serial
import torch
from ultralytics import YOLO

# Silence the "half is deprecated" message (FP16 still works and stays on)
warnings.filterwarnings("ignore", message=".*half.*")


class _NoHalfWarning(logging.Filter):
    def filter(self, record):
        return "half" not in record.getMessage()


logging.getLogger("ultralytics").addFilter(_NoHalfWarning())


GREEN = (0, 200, 0)
RED = (0, 0, 230)
BLUE = (255, 160, 0)
AMBER = (0, 170, 255)
WHITE = (255, 255, 255)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
class Camera:
    """Background camera reader. Always keeps the newest frame."""

    def __init__(self, device=0, width=640, height=480, fps=30):
        pipeline = (
            f"v4l2src device=/dev/video{device} do-timestamp=true ! "
            f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
            "jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! "
            "video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            print("GStreamer failed, using V4L2...")
            self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{device}")

        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


# ---------------------------------------------------------------------------
# Temporal validation of the vision decision
# ---------------------------------------------------------------------------
class TemporalValidator:
    """A decision must stay the same for hold_s seconds before it is accepted.
    Any change during that time restarts the timer."""

    def __init__(self, hold_s=3.0, instant=()):
        self.hold_s = hold_s
        self.instant = set(instant)
        self.confirmed = "UNKNOWN"
        self.candidate = None
        self.since = 0.0

    def update(self, raw):
        now = time.monotonic()
        if raw == self.confirmed:
            self.candidate = None
            return self.confirmed, 0.0
        if raw in self.instant:
            self.confirmed, self.candidate = raw, None
            return self.confirmed, 0.0
        if raw != self.candidate:
            self.candidate, self.since = raw, now
        elapsed = now - self.since
        if elapsed >= self.hold_s:
            self.confirmed, self.candidate = raw, None
            return self.confirmed, 0.0
        return self.confirmed, self.hold_s - elapsed


# ---------------------------------------------------------------------------
# Arduino link with feedback
# ---------------------------------------------------------------------------
class GateLink:
    def __init__(self, port, baudrate, keepalive_s=1.0):
        self.ser = serial.Serial(port, baudrate, timeout=0, write_timeout=1)
        self.keepalive_s = keepalive_s
        self.buf = b""

        self.last_cmd = None        # last command sent
        self.cmd_changed_at = 0.0   # when last_cmd last changed
        self.last_sent = 0.0
        self.acked_cmd = None       # last command the Arduino acknowledged
        self.gate = "UNKNOWN"       # OPEN / CLOSED from the LDR
        self.last_heard = 0.0       # last time any message arrived
        self.ready = False

        # Opening the port resets most Arduinos: wait for "READY"
        print("Waiting for Arduino READY...")
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and not self.ready:
            self.poll()
            time.sleep(0.05)
        print("Arduino READY" if self.ready else "WARNING: no READY received (continuing)")

    def send(self, cmd, force=False):
        now = time.monotonic()
        changed = cmd != self.last_cmd
        if not (changed or force or now - self.last_sent >= self.keepalive_s):
            return
        self.ser.write((cmd + "\n").encode())
        self.last_sent = now
        if changed:
            self.last_cmd = cmd
            self.cmd_changed_at = now
            print(f"UART TX: {cmd}")

    def poll(self):
        """Read and parse everything the Arduino has sent."""
        waiting = self.ser.in_waiting
        if waiting:
            self.buf += self.ser.read(waiting)
        while b"\n" in self.buf:
            raw, self.buf = self.buf.split(b"\n", 1)
            line = raw.decode(errors="ignore").strip()
            if line:
                self._handle(line)

    def _handle(self, line):
        self.last_heard = time.monotonic()
        key, _, value = line.partition(":")

        if key == "READY":
            self.ready = True
            self.acked_cmd = None
        elif key == "ACK":
            if value != self.acked_cmd:
                print(f"UART RX: ACK {value}")
            self.acked_cmd = value
        elif key == "GATE":
            if value != self.gate:
                print(f"UART RX: gate {value}")
            self.gate = value
        elif key == "EVENT":
            print(f"UART RX: event {value}")
        elif key == "ERR":
            print(f"UART RX: Arduino rejected '{value}'")

    def alive(self, timeout_s=2.0):
        return time.monotonic() - self.last_heard < timeout_s

    def close(self):
        try:
            self.ser.write(b"STOP\n")
            self.ser.flush()
            print("UART TX: STOP (shutdown)")
        except Exception:
            pass
        self.ser.close()


# ---------------------------------------------------------------------------
# Decision based on Arduino feedback
# ---------------------------------------------------------------------------
class GateSupervisor:
    """Checks that the gate actually does what it was told."""

    def __init__(self, gate_timeout_s=2.5, ack_timeout_s=1.0, retry_after_s=5.0):
        self.gate_timeout_s = gate_timeout_s
        self.ack_timeout_s = ack_timeout_s
        self.retry_after_s = retry_after_s
        self.block_go_until = 0.0
        self.fault = None

    def decide(self, vision_cmd, link):
        """Returns the command to send, after applying feedback rules."""
        now = time.monotonic()

        # GO is blocked for a while after the gate failed to open
        if vision_cmd == "GO" and now < self.block_go_until:
            return "STOP"
        return vision_cmd

    def check(self, link):
        """Updates self.fault from the feedback. Returns True if STOP must be forced."""
        now = time.monotonic()
        since_change = now - link.cmd_changed_at
        expected = "OPEN" if link.last_cmd == "GO" else "CLOSED"

        if not link.alive():
            self.fault = "ARDUINO NOT RESPONDING"
            return False

        if link.acked_cmd != link.last_cmd and since_change > self.ack_timeout_s:
            self.fault = "NO ACK - RESENDING"     # keep-alive resends it every second
            return False

        if link.gate in ("OPEN", "CLOSED") and link.gate != expected \
                and since_change > self.gate_timeout_s:
            if expected == "OPEN":
                self.fault = "GATE DID NOT OPEN"
                self.block_go_until = now + self.retry_after_s
                return True                       # fail safe: command closed
            self.fault = "GATE DID NOT CLOSE"     # critical: keep sending STOP
            return True

        if now >= self.block_go_until:
            self.fault = None
        return False


def access_text(link):
    if link.last_cmd == "GO":
        return ("ACCESS GRANTED", GREEN) if link.gate == "OPEN" else ("OPENING...", AMBER)
    if link.last_cmd == "STOP":
        return ("ACCESS DENIED", RED) if link.gate == "CLOSED" else ("CLOSING...", AMBER)
    return ("WAITING FOR PERSON", WHITE)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def vest_inside_person(person, vest, threshold=0.5):
    x0, y0 = max(person[0], vest[0]), max(person[1], vest[1])
    x1, y1 = min(person[2], vest[2]), min(person[3], vest[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    vest_area = max(1e-6, (vest[2] - vest[0]) * (vest[3] - vest[1]))
    return intersection / vest_area >= threshold


def draw_label(frame, text, x, y, color):
    y = max(y, 50)
    cv2.putText(frame, text, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)


def put(frame, text, y, color=WHITE, scale=0.45):
    cv2.putText(frame, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best.pt")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--show-vests", action="store_true")
    ap.add_argument("--uart-port", default="/dev/ttyACM0")
    ap.add_argument("--baudrate", type=int, default=9600)
    ap.add_argument("--hold", type=float, default=3.0, help="temporal validation time (s)")
    ap.add_argument("--instant-stop", action="store_true", help="STOP skips the hold time")
    ap.add_argument("--gate-timeout", type=float, default=2.5,
                    help="time allowed for the gate to reach the commanded state (s)")
    args = ap.parse_args()

    camera = None
    link = None
    total_frames = 0
    start = time.perf_counter()

    try:
        print("================================")
        print(f"JETSON PPE - {args.imgsz}x{args.imgsz} + UART FEEDBACK")
        print(f"Temporal validation: {args.hold:.1f}s"
              + (" (STOP instant)" if args.instant_stop else ""))
        print("================================")

        if torch.cuda.is_available():
            device, use_half = 0, True
            print("GPU:", torch.cuda.get_device_name(0), "| CUDA:", torch.version.cuda)
            torch.backends.cudnn.benchmark = True
        else:
            device, use_half = "cpu", False
            print("WARNING: CUDA unavailable")

        print(f"Loading model: {args.model}")
        model = YOLO(args.model, task="detect")

        print(f"Connecting to Arduino: {args.uart_port}")
        link = GateLink(args.uart_port, args.baudrate)

        camera = Camera(device=args.camera, width=args.width, height=args.height, fps=30)
        print("Waiting for camera...")
        while True:
            ok, frame = camera.read()
            if ok:
                break
            time.sleep(0.01)

        print("Warming up...")
        with torch.inference_mode():
            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                   half=use_half, device=device, verbose=False)[0]

        class_ids = {name: cid for cid, name in result.names.items()}
        for needed in ("person", "safety_vest"):
            if needed not in class_ids:
                raise RuntimeError(f"Model has no '{needed}' class. Available: {list(class_ids)}")
        person_id, vest_id = class_ids["person"], class_ids["safety_vest"]

        validator = TemporalValidator(args.hold, ("STOP",) if args.instant_stop else ())
        supervisor = GateSupervisor(gate_timeout_s=args.gate_timeout)

        fps, fps_count, fps_start = 0.0, 0, time.perf_counter()
        print("Running. Press Q or ESC to quit.\n")

        while True:
            link.poll()

            ok, frame = camera.read()
            if not ok:
                continue
            frame = frame.copy()

            with torch.inference_mode():
                result = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                       half=use_half, device=device, verbose=False)[0]

            persons, vests = [], []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().tolist()
                classes = result.boxes.cls.int().cpu().tolist()
                for box, cls in zip(boxes, classes):
                    if cls == person_id:
                        persons.append(box)
                    elif cls == vest_id:
                        vests.append(box)

            n_vest = n_no_vest = 0
            for person in persons:
                wearing = any(vest_inside_person(person, v, 0.5) for v in vests)
                color = GREEN if wearing else RED
                n_vest += wearing
                n_no_vest += not wearing
                x0, y0, x1, y1 = map(int, person)
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                draw_label(frame, "VEST" if wearing else "NO VEST", x0, y0, color)

            if args.show_vests:
                for v in vests:
                    x0, y0, x1, y1 = map(int, v)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), BLUE, 1)

            # ---- Decision: vision -> validation -> feedback rules -> Arduino ----
            if not persons:
                raw_cmd = "UNKNOWN"
            elif n_no_vest > 0:
                raw_cmd = "STOP"
            else:
                raw_cmd = "GO"

            vision_cmd, remaining = validator.update(raw_cmd)
            command = supervisor.decide(vision_cmd, link)
            link.send(command)
            link.poll()
            if supervisor.check(link):
                link.send("STOP")

            # ---- FPS ----
            total_frames += 1
            fps_count += 1
            now = time.perf_counter()
            if now - fps_start >= 1.0:
                fps = fps_count / (now - fps_start)
                fps_count, fps_start = 0, now

            # ---- Display ----
            w = frame.shape[1]
            cv2.rectangle(frame, (0, 0), (w, 64), (0, 0, 0), -1)

            pending = f" | validating {validator.candidate} {remaining:.1f}s" if remaining > 0 else ""
            put(frame, f"People: {len(persons)} | Vest: {n_vest} | No vest: {n_no_vest}"
                       f"{pending} | FPS: {fps:.1f}", 18)

            ack = link.acked_cmd or "-"
            put(frame, f"Sent: {link.last_cmd or '-'} | ACK: {ack} | Gate (LDR): {link.gate}", 38)

            text, color = access_text(link)
            if supervisor.fault:
                text, color = f"FAULT: {supervisor.fault}", RED
            put(frame, text, 58, color, 0.5)

            cv2.imshow("PPE Detection - UART feedback", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        if link is not None:
            link.close()
        if camera is not None:
            camera.stop()
        cv2.destroyAllWindows()

        elapsed = time.perf_counter() - start
        print("\n================================")
        print("Finished")
        print("================================")
        print(f"Frames: {total_frames}")
        print(f"Average FPS: {total_frames / max(elapsed, 1e-6):.1f}")


if __name__ == "__main__":
    main()
