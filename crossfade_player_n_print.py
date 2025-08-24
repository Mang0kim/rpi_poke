#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video player (OpenCV) + HX711 (gandalf15/HX711)
- Zero-based: weight = A * (EMA(raw) - zero_raw)
- Start auto-zero, re-zero with 'Z'
- Adaptive EMA: 큰 변화는 빠르게, 잔잔할 때는 부드럽게
- Zero-Lock: 0 근처에서 빠르게 0으로 스냅(영점 고정)
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

# Calibration slope (two-point fit)
A = 0.0355646605  # grams per raw-count

# Sampling / smoothing
ZERO_SAMPLES = 60      # 자동/재영점 시 평균 샘플 수
READ_SAMPLES = 12      # 평상시 읽기 평균 샘플 수

# --- Adaptive EMA (잔잔할 땐 SLOW, 큰 변동은 FAST) ---
EMA_ALPHA_SLOW   = 0.12   # 0~1, 작을수록 부드럽게
EMA_ALPHA_FAST   = 0.50   # 큰 변동일 때 빨리 따라감
DELTA_RAW_FAST   = 400    # |raw - ema| >= 이면 FAST 사용

# 출력/루프
PRINT_EVERY = 0.2         # s
LOOP_SLEEP  = 0.04        # s (~25Hz)

# --- Zero-Lock (0 근처에서 빠른 복귀) ---
ZERO_LOCK_THRESHOLD = 5.0   # g, 이 범위 이내면 '거의 0'
ZERO_LOCK_MIN_TIME  = 0.4   # s, 이 시간 연속 유지되면 zero_raw를 EMA로 스냅

# -------------------------------
# Control flags / shared state
# -------------------------------
stop_event   = threading.Event()
zero_request = threading.Event()
state_lock   = threading.Lock()   # zero_raw 보호

zero_raw = None  # 전역 영점(raw), 시작/재영점 시 갱신

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
    """빈하중 raw 평균 리턴(None 가능)."""
    return hx.get_raw_data_mean(ZERO_SAMPLES)

def _auto_zero(tag="start") -> bool:
    """현재 빈하중으로 zero_raw 설정."""
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

    # get_data_mean() 용 offset 정렬(우리는 raw 사용하지만 내부 일관성용)
    off = hx.get_raw_data_mean(20)
    if off is not None:
        hx.set_offset(off)
        print(f"[HX711] offset (for get_data_mean) set to {off}")

    # 시작 자동 영점
    if not _auto_zero("start"):
        with state_lock:
            zero_raw = off if off is not None else 0

    smoothed_raw   = None
    zero_win_start = None
    last_print     = 0.0

    try:
        while not stop_event.is_set():
            # 재영점 처리
            if zero_request.is_set():
                time.sleep(0.5)      # 무게 제거 시간
                if _auto_zero("re-zero"):
                    smoothed_raw = None   # 이전 EMA 편향 제거
                    zero_win_start = None
                zero_request.clear()

            raw = hx.get_raw_data_mean(READ_SAMPLES)
            if raw is not None:
                # --- Adaptive EMA ---
                if smoothed_raw is None:
                    smoothed_raw = raw
                else:
                    delta = abs(raw - smoothed_raw)
                    alpha = EMA_ALPHA_FAST if delta >= DELTA_RAW_FAST else EMA_ALPHA_SLOW
                    smoothed_raw = alpha * raw + (1.0 - alpha) * smoothed_raw

                # 무게 계산
                with state_lock:
                    zr = zero_raw
                weight = A * (smoothed_raw - zr)

                now = time.time()

                # --- Zero-Lock: 0 근처에서 빠른 복귀 ---
                if abs(weight) < ZERO_LOCK_THRESHOLD:
                    if zero_win_start is None:
                        zero_win_start = now
                    elif (now - zero_win_start) >= ZERO_LOCK_MIN_TIME:
                        with state_lock:
                            zero_raw = smoothed_raw  # 영점 스냅
                        zero_win_start = None
                        print("[CAL] zero-lock: zero_raw snapped to EMA for fast return")
                else:
                    zero_win_start = None

                # 출력
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
