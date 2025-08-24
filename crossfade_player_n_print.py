#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video player (OpenCV) + HX711 (gandalf15/HX711)
- weight = A * (EMA(raw) - zero_raw)
- Start auto-zero, re-zero with 'Z'
- Adaptive EMA (fast on large changes)
- Zero-Lock: near-zero snap for fast return
- One-key gain calibration: press 'C' with a known mass on the scale
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
VIDEO_PATH   = "01.mp4"
WINDOW_NAME  = "Player"
HEADLESS     = False
SPEED_SCALE  = 1.0

# Calibration slope (initial; can be re-calibrated with 'C')
A = 0.0355646605  # grams per raw-count

# Known mass for gain calibration (in grams)
CAL_MASS_G = 69200.0  # 예: 69.2 kg

# Sampling / smoothing
ZERO_SAMPLES = 60      # auto/re-zero averaging
READ_SAMPLES = 12      # per-read averaging

# Adaptive EMA
EMA_ALPHA_SLOW = 0.12  # steady state
EMA_ALPHA_FAST = 0.50  # when big jump
DELTA_RAW_FAST = 400   # |raw - ema| >= -> FAST

# Printing / loop timing
PRINT_EVERY = 0.2      # seconds
LOOP_SLEEP  = 0.04     # seconds (~25 Hz)

# Zero-Lock
ZERO_LOCK_THRESHOLD = 5.0  # g
ZERO_LOCK_MIN_TIME  = 0.4  # s

# -------------------------------
# Control flags / shared state
# -------------------------------
stop_event    = threading.Event()
zero_request  = threading.Event()
cal_request   = threading.Event()  # 'C' pressed -> gain calibration
state_lock    = threading.Lock()   # protects zero_raw and A

zero_raw = None  # updated on (re)zero

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
    """Averaged empty-load raw (or None)."""
    return hx.get_raw_data_mean(ZERO_SAMPLES)

def _auto_zero(tag="start") -> bool:
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
            elif key in (ord('c'), ord('C')):
                print(f"[CAL] C pressed -> gain calibration request (mass={CAL_MASS_G} g)")
                cal_request.set()
        else:
            time.sleep(delay_ms / 1000.0)

    cap.release()
    return True

# -------------------------------
# HX711 Reader Thread
# -------------------------------
def hx711_reader():
    global zero_raw, A
    hx.reset()

    # Align get_data_mean() offset (we use raw, but keeps lib consistent)
    off = hx.get_raw_data_mean(20)
    if off is not None:
        hx.set_offset(off)
        print(f"[HX711] offset (for get_data_mean) set to {off}")

    # Start auto-zero
    if not _auto_zero("start"):
        with state_lock:
            zero_raw = off if off is not None else 0

    smoothed_raw   = None
    zero_win_start = None
    last_print     = 0.0

    try:
        while not stop_event.is_set():
            # Re-zero request
            if zero_request.is_set():
                time.sleep(0.5)  # time to clear the scale
                if _auto_zero("re-zero"):
                    smoothed_raw = None
                    zero_win_start = None
                zero_request.clear()

            raw = hx.get_raw_data_mean(READ_SAMPLES)
            if raw is not None:
                # Adaptive EMA
                if smoothed_raw is None:
                    smoothed_raw = raw
                else:
                    delta = abs(raw - smoothed_raw)
                    alpha = EMA_ALPHA_FAST if delta >= DELTA_RAW_FAST else EMA_ALPHA_SLOW
                    smoothed_raw = alpha * raw + (1.0 - alpha) * smoothed_raw

                # Compute weight
                with state_lock:
                    zr = zero_raw
                    curA = A
                weight = curA * (smoothed_raw - zr)

                now = time.time()

                # Zero-Lock (snap to zero when near 0 for a short time)
                if abs(weight) < ZERO_LOCK_THRESHOLD:
                    if zero_win_start is None:
                        zero_win_start = now
                    elif (now - zero_win_start) >= ZERO_LOCK_MIN_TIME:
                        with state_lock:
                            zero_raw = smoothed_raw
                        zero_win_start = None
                        print("[CAL] zero-lock: zero_raw snapped to EMA for fast return")
                else:
                    zero_win_start = None

                # Gain calibration (press 'C' with known mass on the scale)
                if cal_request.is_set():
                    with state_lock:
                        zr_local = zero_raw
                    delta_counts = smoothed_raw - zr_local
                    if abs(delta_counts) > 1.0:
                        newA = CAL_MASS_G / float(delta_counts)
                        with state_lock:
                            oldA = A
                            A = newA
                        print(f"[CAL] gain: delta={int(delta_counts)}, A(old)={oldA:.8f} -> A(new)={newA:.8f}")
                    else:
                        print("[CAL] gain: delta too small; put the mass on the scale.")
                    cal_request.clear()

                # Print
                if (now - last_print) >= PRINT_EVERY:
                    print(f"[HX711] raw={int(raw)}, ema={int(smoothed_raw)}, zero_raw={int(zr)}, weight={weight:.2f}")
                    last_print = now
            else:
                print("[HX711] invalid data")

            time.sleep(LOOP_SLEEP)
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
