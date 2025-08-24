#!/usr/bin/env python3
# -------------------------------
# 1) Import required libraries
# -------------------------------
import cv2
import sys
import time
import RPi.GPIO as GPIO
from hx711_simple import HX711Simple

GPIO.setmode(GPIO.BCM)

# -------------------------------
# 2) Global settings / variables
# -------------------------------
# video
WINDOW_NAME = "Player"
HEADLESS = False        # True = no display
RESPECT_TIMING = True   # Match original FPS even in headless
SPEED_SCALE = 1.0       # 1.0 = normal, 0.5 = half speed, 2.0 = double

hx = HX711Simple(dout_pin=5, sck_pin=6, channel_select = "A", channel_A_gain=128)

# -------------------------------
# 3) Functions
# -------------------------------
def _calc_delay_ms(cap) -> int:
    """Compute frame delay (ms) from video FPS with fallback."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0  # fallback
    delay = max(1, int(round(1000.0 / fps)))
    # Apply speed scale (e.g., 0.5 => slower => larger delay)
    delay = max(1, int(round(delay / max(1e-6, SPEED_SCALE))))
    return delay

def play(path: str) -> bool:
    """Generic video play function (FPS-synced)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {path}")
        return False

    delay_ms = _calc_delay_ms(cap)

    if not HEADLESS:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not HEADLESS:
            cv2.imshow(WINDOW_NAME, frame)
            # Esc to quit early
            if cv2.waitKey(delay_ms) & 0xFF == 27:
                cap.release()
                return True
        else:
            # Headless timing control
            if RESPECT_TIMING:
                time.sleep(delay_ms / 1000.0)
            # If not respecting timing, loop as fast as possible

    cap.release()
    return True

def play01(): return play("01.mp4")
def play02(): return play("02.mp4")
def play03(): return play("03.mp4")
def play03_1(): return play("03-1.mp4")

def cleanup():
    if not HEADLESS:
        cv2.destroyAllWindows()

# -------------------------------
# 4) Main loop
# -------------------------------
if __name__ == "__main__":
    try:
        while True:
            play01()
            play02()

            try:
                value = hx.read_signed()
                print("Weight raw:", value)
            finally:
                hx.cleanup()

            play03()
            play03_1()
            break  # remove for endless loop
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
        sys.exit(0)
