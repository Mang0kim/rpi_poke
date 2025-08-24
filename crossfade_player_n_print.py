#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video player (OpenCV) + HX711 reader (gandalf15/HX711)
"""

import cv2
import time
import threading
import signal
import sys

from hx711 import HX711  # gandalf15/HX711 라이브러리

# -------------------------------
# USER SETTINGS
# -------------------------------
VIDEO_PATH = "01.mp4"
WINDOW_NAME = "Player"
HEADLESS = False
SPEED_SCALE = 1.0

# HX711 핀 (BCM numbering)
DOUT_PIN = 5   # 예시: GPIO5
SCK_PIN = 6    # 예시: GPIO6

# 무게 변환 식
A = 0.03557
B = 7059.6

PRINT_EVERY = 0.2  # seconds

# -------------------------------
# CONTROL FLAG
# -------------------------------
stop_event = threading.Event()


def _install_sig_handlers():
    def _handler(sig, frame):
        stop_event.set()
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# -------------------------------
# Video Player
# -------------------------------
def _calc_delay_ms(cap) -> int:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0
    delay = max(1, int(round(1000.0 / fps)))
    delay = max(1, int(round(delay / max(1e-6, SPEED_SCALE))))
    return delay


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
        else:
            time.sleep(delay_ms / 1000.0)

    cap.release()
    return True


# -------------------------------
# HX711 Reader Thread
# -------------------------------
def hx711_reader():
    hx = HX711(DOUT_PIN, SCK_PIN)
    hx.reset()
    hx.tare()  # 초기 영점 맞춤

    last_print = 0.0
    try:
        while not stop_event.is_set():
            raw = hx.get_raw_data_mean(10)  # raw 평균값
            if raw is not None:
                weight = A * raw + B
                now = time.time()
                if now - last_print >= PRINT_EVERY:
                    print(f"[HX711] raw={raw}, weight={weight:.2f}")
                    last_print = now
            else:
                print("[HX711] invalid data")
            time.sleep(0.05)
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
        sys.exit(0)


if __name__ == "__main__":
    main()
