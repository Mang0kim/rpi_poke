#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2, time, threading, signal, sys
from hx711 import HX711
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
hx = HX711(dout_pin=5, pd_sck_pin=6)

VIDEO_PATH = "01.mp4"
WINDOW_NAME = "Player"
HEADLESS = False
SPEED_SCALE = 1.0

# --- Calibration ---
A = 0.0355646605
RAW_ZERO = -209727           # 새로 측정한 빈하중 raw
B = -A * RAW_ZERO            # 자동 계산된 절편(≈ 7458.87)

PRINT_EVERY = 0.2
stop_event = threading.Event()

def _install_sig_handlers():
    def _h(sig, frame): stop_event.set()
    signal.signal(signal.SIGINT, _h); signal.signal(signal.SIGTERM, _h)

def _calc_delay_ms(cap)->int:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1e-3: fps = 30.0
    d = max(1, int(round(1000.0 / fps)))
    return max(1, int(round(d / max(1e-6, SPEED_SCALE))))

def play_video(path: str)->bool:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {path}"); return False
    delay_ms = _calc_delay_ms(cap)
    if not HEADLESS:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
        if not HEADLESS:
            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(delay_ms) & 0xFF) == 27:  # ESC
                stop_event.set(); break
        else:
            time.sleep(delay_ms/1000.0)
    cap.release(); return True

def hx711_reader():
    hx.reset()
    # 참고: set_offset은 get_data_mean()에 주 영향. 우리는 get_raw_data_mean() 사용.
    offset = hx.get_raw_data_mean(20)
    if offset is not None:
        hx.set_offset(offset)
        print(f"[HX711] offset set to {offset}")
    else:
        print("[HX711] warning: could not set offset")

    print(f"[CAL] Using A={A:.9f}, RAW_ZERO={RAW_ZERO}, B={B:.4f}")

    last_print = 0.0
    try:
        while not stop_event.is_set():
            raw = hx.get_raw_data_mean(10)
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

def main():
    _install_sig_handlers()
    t = threading.Thread(target=hx711_reader, daemon=True); t.start()
    try:
        play_video(VIDEO_PATH)
    finally:
        stop_event.set()
        if not HEADLESS:
            try: cv2.destroyAllWindows()
            except Exception: pass
        t.join(timeout=2.0); sys.exit(0)

if __name__ == "__main__":
    main()
