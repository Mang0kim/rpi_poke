#ver10
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video player (OpenCV) + HX711 (gandalf15/HX711)
- Zero-based conversion: weight = A * (EMA(raw) - zero_raw)
- Auto zero at start, re-zero with 'Z'
- EMA filter to reduce jitter
"""

import cv2
import time
import threading
import signal
import sys
from hx711 import HX711
import RPi.GPIO as GPIO

# -------------------------------
# GPIO / HX711
# -------------------------------
GPIO.setmode(GPIO.BCM)
hx = HX711(dout_pin=5, pd_sck_pin=6)  # DOUT=GPIO5, SCK=GPIO6

# -------------------------------
# USER SETTINGS
# -------------------------------
VIDEO_PATH = "01.mp4"
WINDOW_NAME = "Player"
HEADLESS = False
SPEED_SCALE = 1.0

# Calibration slope (from your two-point fit)
A = 0.0355646605  # grams per raw-count

# Sampling / smoothing
ZERO_SAMPLES = 60      # samples used to compute zero_raw
READ_SAMPLES = 12      # samples per read for mean
EMA_ALPHA = 0.12       # 0~1, smaller -> smoother
PRINT_EVERY = 0.2      # seconds

# -------------------------------
# Control flags / shared state
# -------------------------------
stop_event = threading.Event()
zero_request = threading.Event()
state_lock = threading.Lock()   # protects zero_raw

zero_raw = None                 # updated at auto-zero / re-zero

def _install_sig_handlers():
    def _handler(sig, frame):
        stop_event.set()
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

# -------------------------------
# Helpers
# -------------------------------
def _calc_delay_ms(cap) -> int:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0
    delay = max(1, int(round(1000.0 / fps)))
    delay = max(1, int(round(delay / max(1e-6, SPEED_SCALE))))
    return delay

def _measure_zero_raw():
    """Return averaged raw at empty load (or None)."""
    return hx.get_raw_data_mean(ZERO_SAMPLES)

def _auto_zero(tag="start"):
    """Set zero_raw from current empty-load measurement."""
    global zero_raw
    raw0 = _measure_zero_raw()
    if raw0 is None:
        print(f"[CAL] {tag}: zero measurement failed")
        return False
    with state_lock:
        zero_raw = raw0
    print(f"[CAL] {tag}: zero_raw set to {raw0}")
    return True

# -------------------------------
# Video Player
# -------------------------------
def play_video(path: str) -> bool:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {path}")
        return False

    delay_ms = _calc_delay_ms(cap)

    if not HEADLESS:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
            continue

        if not HEADLESS:
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == 27:  # ESC
                stop_event.set()
                break
            elif key in (ord('z'), ord('Z')):
                print("[CAL] Z pressed -> re-zero request (remove all weight)")
                zero_request.set()
        else:
            time.sleep(delay_ms / 1000.0)

    cap.release()
    return True

# -------------------------------
# HX711 Reader Thread
# -------------------------------
def hx711_reader():
    global zero_raw
    
    hx.reset()

    # (Optional) align get_data_mean() offset, though we use get_raw_data_mean()
    off = hx.get_raw_data_mean(20)
    if off is not None:
        hx.set_offset(off)
        print(f"[HX711] offset (for get_data_mean) set to {off}")

    # Auto zero at start
    if not _auto_zero("start"):
        # fallback: take whatever we have to avoid None
        with state_lock:
            zero_raw = off if off is not None else 0

    smoothed_raw = None
    last_print = 0.0

    try:
        while not stop_event.is_set():
            # Handle re-zero request
            if zero_request.is_set():
                time.sleep(0.5)   # give operator time to clear the scale
                if _auto_zero("re-zero"):
                    smoothed_raw = None    # reset EMA to avoid bias carryover
                zero_request.clear()

            raw = hx.get_raw_data_mean(READ_SAMPLES)
            if raw is not None:
                # EMA filter
                if smoothed_raw is None:
                    smoothed_raw = raw
                else:
                    smoothed_raw = EMA_ALPHA * raw + (1.0 - EMA_ALPHA) * smoothed_raw

                with state_lock:
                    zr = zero_raw

                weight = A * (smoothed_raw - zr)

                now = time.time()
                if now - last_print >= PRINT_EVERY:
                    print(f"[HX711] raw={int(raw)}, ema={int(smoothed_raw)}, zero_raw={int(zr)}, weight={weight:.2f}")
                    last_print = now
            else:
                print("[HX711] invalid data")

            time.sleep(0.04)  # ~25Hz
    except Exception as e:
        print("[HX711] exception:", e)

# -------------------------------
# MAIN
# -------------------------------
def main():
    _install_sig_handlers()

    t = threading.Thread(target=hx711_reader, daemon=True)
    t.start()

    try:
        play_video(VIDEO_PATH)
    finally:
        stop_event.set()
        if not HEADLESS:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        t.join(timeout=2.0)
        GPIO.cleanup()
        sys.exit(0)

if __name__ == "__main__":
    main()
