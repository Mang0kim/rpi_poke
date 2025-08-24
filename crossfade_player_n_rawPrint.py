#!/usr/bin/env python3
# -------------------------------
# 1) Import required libraries
# -------------------------------
import cv2
import sys
import time
try:
    from HX711 import HX711  # pip install HX711
except Exception as e:
    HX711 = None
    print("[Warn] HX711 library not found:", e)

# -------------------------------
# 2) Global settings / variables
# -------------------------------
WINDOW_NAME = "Player"
HEADLESS = False        # True = no display
RESPECT_TIMING = True   # Match original FPS even in headless
SPEED_SCALE = 1.0       # 1.0 = normal, 0.5 = half speed, 2.0 = double

# HX711 pins (change to your wiring)
DOUT_PIN = 5            # GPIO pin number for DOUT
SCK_PIN  = 6            # GPIO pin number for SCK

# -------------------------------
# 3) Functions
# -------------------------------
def _calc_delay_ms(cap) -> int:
    """Compute frame delay (ms) from video FPS with fallback."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0
    delay = max(1, int(round(1000.0 / fps)))
    delay = max(1, int(round(delay / max(1e-6, SPEED_SCALE))))
    return delay

def play(path: str):
    """Play a video once (FPS-synced). Return the last frame (or None)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {path}")
        return None

    delay_ms = _calc_delay_ms(cap)

    if not HEADLESS:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    last_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame
        if not HEADLESS:
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(delay_ms) & 0xFF == 27:
                cap.release()
                return last_frame
        else:
            if RESPECT_TIMING:
                time.sleep(delay_ms / 1000.0)

    cap.release()
    return last_frame

def play01(): return play("01.mp4")
def play02(): return play("02.mp4")
def play03(): return play("03.mp4")
def play03_1(): return play("03-1.mp4")

def read_hx711_raw(samples: int = 10, sample_delay: float = 0.01) -> int:
    """Read raw data from HX711; return integer mean of samples."""
    if HX711 is None:
        print("[Error] HX711 library unavailable.")
        return 0
    hx = HX711(dout_pin=DOUT_PIN, pd_sck_pin=SCK_PIN)
    hx.reset()
    hx.power_up()
    # Use library helper if available; otherwise simple average
    try:
        val = hx.get_raw_data_mean(readings=samples)
    except Exception:
        vals = []
        for _ in range(samples):
            vals.append(hx.get_raw_data())
            time.sleep(sample_delay)
        val = int(sum(vals) / max(1, len(vals)))
    hx.power_down()
    return int(val) if val is not None else 0

def freeze_and_measure(frame) -> None:
    """Show the given frame frozen, measure HX711 once, print raw, then continue."""
    # Show frozen frame if display is enabled
    if not HEADLESS and frame is not None:
        cv2.imshow(WINDOW_NAME, frame)
        # Small wait to ensure the frame is actually drawn
        cv2.waitKey(1)

    # Read HX711 raw once (adjust samples if needed)
    raw = read_hx711_raw(samples=15, sample_delay=0.005)
    print(f"HX711 raw: {raw}")

    # Keep the frozen frame visible a short moment (optional)
    if not HEADLESS and frame is not None:
        cv2.waitKey(300)  # 300 ms pause on frozen frame

def cleanup():
    """Release all windows."""
    if not HEADLESS:
        cv2.destroyAllWindows()

# -------------------------------
# 4) Main loop
# -------------------------------
if __name__ == "__main__":
    try:
        while True:
            play01()
            last2 = play02()       # we need the final frame of 02.mp4
            freeze_and_measure(last2)
            play03()
            play03_1()
            break  # remove for endless loop
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
        sys.exit(0)
