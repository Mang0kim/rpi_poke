#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HX711 + OpenCV Sequencer
- Keeps adaptive EMA + Zero-Lock + Z/C controls
- Sequences:
  01(wait) -> 02(measure twice, avg) -> 03/03-1(result pause/resume) -> 01
"""

import cv2, time, threading, signal, sys
from hx711 import HX711
import RPi.GPIO as GPIO

# -------------------------------
# HX711 / GPIO
# -------------------------------
GPIO.setmode(GPIO.BCM)
hx = HX711(dout_pin=5, pd_sck_pin=6)  # DOUT=5, SCK=6

# -------------------------------
# USER SETTINGS
# -------------------------------
WINDOW_NAME   = "Player"
HEADLESS      = False
SPEED_SCALE   = 1.0

# Calibration slope (press 'C' to recalibrate with known mass)
A = 0.03716
CAL_MASS_G = 69200.0

# Sampling / smoothing
ZERO_SAMPLES  = 60
READ_SAMPLES  = 8
EMA_ALPHA_SLOW = 0.18
EMA_ALPHA_FAST = 0.60
DELTA_RAW_FAST = 250
PRINT_EVERY    = 0.5
LOOP_SLEEP     = 0.12

# Zero-Lock
ZERO_LOCK_THRESHOLD = 5.0  # g
ZERO_LOCK_MIN_TIME  = 0.3  # s

# Sequence thresholds
TRIGGER_MIN  = 20000.0      # >=
TRIGGER_MAX  = 120000.0     # <
BRANCH_SPLIT = 65000.0      # seq02 avg split
PAUSE_UNDER  = 20000.0      # < for 2 seconds
PAUSE_HOLD_S = 2.0

# -------------------------------
# Shared state / events
# -------------------------------
stop_event    = threading.Event()
zero_request  = threading.Event()
cal_request   = threading.Event()
state_lock    = threading.Lock()

zero_raw = None          # updated by (re)zero
cur_weight_g = 0.0       # latest filtered weight (for UI/logic)
curA = A                 # safe copy for printing

def _install_sig_handlers():
    def _h(sig, frame): stop_event.set()
    signal.signal(signal.SIGINT, _h); signal.signal(signal.SIGTERM, _h)

# ------------- HX711 helpers -------------
def _measure_zero_raw():
    return hx.get_raw_data_mean(ZERO_SAMPLES)

def _auto_zero(tag="start") -> bool:
    global zero_raw
    raw0 = _measure_zero_raw()
    if raw0 is None:
    print(f"[CAL] {tag}: zero measurement failed, keeping last zero_raw={zero_raw}")
    return False
  
    with state_lock:
        zero_raw = raw0
    print(f"[CAL] {tag}: zero_raw set to {raw0}")
    return True

# ------------- Video helpers -------------
def _fps_delay_ms(cap) -> int:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3: fps = 30.0
    d = max(1, int(round(1000.0 / fps)))
    return max(1, int(round(d / max(1e-6, SPEED_SCALE))))

def _show_frame(frame, delay_ms):
    if not HEADLESS:
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == 27:  # ESC
            stop_event.set()
        elif key in (ord('z'), ord('Z')):
            print("[CAL] Z pressed -> re-zero request")
            zero_request.set()
        elif key in (ord('c'), ord('C')):
            print(f"[CAL] C pressed -> gain calibration request (mass={CAL_MASS_G} g)")
            cal_request.set()
    else:
        time.sleep(delay_ms / 1000.0)

def _open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {path}")
        return None, None
    delay_ms = _fps_delay_ms(cap)
    if not HEADLESS:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    return cap, delay_ms

def _play_once(path: str, on_tick=None):
    """
    Play video to last frame (exactly once). on_tick(t_sec, is_last) optional.
    Returns True if finished normally, False if early stop.
    """
    cap, delay_ms = _open_video(path)
    if cap is None: return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    t_sec = 0.0
    is_last = False
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        # check if next read is last: compare positions
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        is_last = (total > 0 and pos >= total)
        if on_tick:
            on_tick(t_sec, is_last)
        _show_frame(frame, delay_ms)
        t_sec += 1.0 / fps
    cap.release()
    return not stop_event.is_set()

def _play_pause_at(path: str, pause_time: float, until_cond):
    """
    Play video; when reaching pause_time (sec), pause on that frame until until_cond() True.
    Then resume to last frame.
    """
    cap, delay_ms = _open_video(path)
    if cap is None: return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    t_sec = 0.0
    paused_frame = None
    # play until pause_time
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return not stop_event.is_set()
        if t_sec + 1e-6 >= pause_time and paused_frame is None:
            paused_frame = frame.copy()
            break
        _show_frame(frame, delay_ms)
        t_sec += 1.0 / fps
    # pause loop
    while not stop_event.is_set() and not until_cond():
        _show_frame(paused_frame, delay_ms)
    # resume to end
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        _show_frame(frame, delay_ms)
    cap.release()
    return not stop_event.is_set()

# -------------------------------
# HX711 reader thread (unchanged core + publish cur_weight_g)
# -------------------------------
def hx711_reader():
    global zero_raw, cur_weight_g, curA, A
    hx.reset()
    off = hx.get_raw_data_mean(20)
    if off is not None:
        hx.set_offset(off)
        print(f"[HX711] offset (for get_data_mean) set to {off}")
    if not _auto_zero("start"):
        with state_lock:
            zero_raw = off if off is not None else 0

    smoothed_raw = None
    zero_win_start = None
    last_print = 0.0

    try:
        while not stop_event.is_set():
            if zero_request.is_set():
                time.sleep(0.5)
                if _auto_zero("re-zero"):
                    smoothed_raw = None; zero_win_start = None
                zero_request.clear()

            raw = hx.get_raw_data_mean(READ_SAMPLES)
            if raw is not None:
                # adaptive EMA
                if smoothed_raw is None:
                    smoothed_raw = raw
                else:
                    delta = abs(raw - smoothed_raw)
                    alpha = EMA_ALPHA_FAST if delta >= DELTA_RAW_FAST else EMA_ALPHA_SLOW
                    smoothed_raw = alpha * raw + (1.0 - alpha) * smoothed_raw

                with state_lock:
                    zr = zero_raw
                    curA = A
                weight = curA * (smoothed_raw - zr)
                cur_weight_g = float(weight)

                now = time.time()
                # zero-lock
                if abs(weight) < ZERO_LOCK_THRESHOLD:
                    if zero_win_start is None:
                        zero_win_start = now
                    elif (now - zero_win_start) >= ZERO_LOCK_MIN_TIME:
                        with state_lock:
                            zero_raw = smoothed_raw
                        zero_win_start = None
                        print("[CAL] zero-lock: zero_raw snapped to EMA")
                else:
                    zero_win_start = None

                # gain calibration on 'C'
                if cal_request.is_set():
                    with state_lock: zr_local = zero_raw
                    delta_counts = smoothed_raw - zr_local
                    if abs(delta_counts) > 1.0:
                        newA = CAL_MASS_G / float(delta_counts)
                        with state_lock:
                            oldA = A; A = newA
                        print(f"[CAL] gain: delta={int(delta_counts)}, A(old)={oldA:.8f} -> A(new)={newA:.8f}")
                    else:
                        print("[CAL] gain: delta too small; put mass on scale.")
                    cal_request.clear()

                if (now - last_print) >= PRINT_EVERY:
                    print(f"[HX711] raw={int(raw)}, ema={int(smoothed_raw)}, zero_raw={int(zr)}, weight={weight:.2f}")
                    last_print = now
            else:
                print("[HX711] invalid data")

            time.sleep(LOOP_SLEEP)
    except Exception as e:
        print("[HX711] exception:", e)

# -------------------------------
# Sequence logic
# -------------------------------
def get_weight():
    return float(cur_weight_g)

def seq01_wait():
    """01.mp4 반복. 트리거 범위 감지되면 그 '회차' 끝까지 재생 후 True 리턴."""
    print("[SEQ] 01(wait) start")
    trigger_armed = False
    while not stop_event.is_set():
        def on_tick(t, is_last):
            nonlocal trigger_armed
            w = get_weight()
            if (TRIGGER_MIN <= w < TRIGGER_MAX):
                trigger_armed = True
        ok = _play_once("01.mp4", on_tick)
        if not ok: return False
        if trigger_armed:
            print("[SEQ] 01 -> 02")
            return True

def seq02_measure():
    """02.mp4 두 번 재생. 처음 2s 버리고 평균. 분기 후 (True, '03'|'03-1'|'01')."""
    print("[SEQ] 02(measure) start")
    start_time = time.time()
    valid_sum = 0.0
    valid_cnt = 0
    def tick(_t, _is_last):
        nonlocal valid_sum, valid_cnt
        elapsed = time.time() - start_time
        if elapsed >= 2.0:  # 버퍼 구간 제외
            valid_sum += get_weight()
            valid_cnt += 1

    # 두 번 재생
    for rep in range(2):
        ok = _play_once("02.mp4", tick)
        if not ok: return False, None

    if valid_cnt == 0:
        avg = 0.0
    else:
        avg = valid_sum / valid_cnt
    print(f"[SEQ] 02 avg={avg:.2f} g (cnt={valid_cnt})")

    if avg < TRIGGER_MIN:
        print("[SEQ] 02 -> 01 (avg < 20kg)")
        return True, "01"
    elif avg < BRANCH_SPLIT:
        print("[SEQ] 02 -> 03 (avg < 65kg)")
        return True, "03"
    else:
        print("[SEQ] 02 -> 03-1 (avg >= 65kg)")
        return True, "03-1"

def seq03_pause_then_resume(video_name):
    """
    03/03-1: 2.0초에서 일시정지. weight<20kg 2초 연속 유지되면 나머지 재생.
    """
    print(f"[SEQ] {video_name} start (pause@2.0s)")
    # 조건 함수: 2초 연속 under
    under_start = [None]  # use list to close over mutable

    def cond():
        w = get_weight()
        now = time.time()
        if w < PAUSE_UNDER:
            if under_start[0] is None:
                under_start[0] = now
            return (now - under_start[0]) >= PAUSE_HOLD_S
        else:
            under_start[0] = None
            return False

    ok = _play_pause_at(video_name, pause_time=2.0, until_cond=cond)
    if not ok: return False
    print(f"[SEQ] {video_name} -> 01")
    return True

def run_sequences():
    while not stop_event.is_set():
        if not seq01_wait(): break
        ok, branch = seq02_measure()
        if not ok: break
        if branch == "01":
            # 바로 루프 계속 -> 01로
            continue
        elif branch == "03":
            if not seq03_pause_then_resume("03.mp4"): break
        elif branch == "03-1":
            if not seq03_pause_then_resume("03-1.mp4"): break
        # 이후 자동으로 01로 루프

# -------------------------------
# MAIN
# -------------------------------
def main():
    _install_sig_handlers()
    t = threading.Thread(target=hx711_reader, daemon=True); t.start()
    try:
        run_sequences()
    finally:
        stop_event.set()
        if not HEADLESS:
            try: cv2.destroyAllWindows()
            except Exception: pass
        t.join(timeout=2.0)
        GPIO.cleanup()
        sys.exit(0)

if __name__ == "__main__":
    main()
