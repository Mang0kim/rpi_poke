#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HX711 + OpenCV Sequencer (10SPS 기준)
- Adaptive EMA + Zero-Lock + Z/C controls
- Safe reader: is_ready() 확인 후 -1/0/None 샘플 제거
- Sequences:
  01(wait ≥1s) -> 02(measure twice + drop<=50% running avg) -> 03/03-1(pause/resume) -> 01
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
CAL_MASS_G = 69200.0  # 예: 69.2 kg

# Sampling / smoothing (10SPS 모드)
ZERO_SAMPLES   = 10     # auto zero 평균 샘플 수
READ_SAMPLES   = 2      # per-read 평균 샘플 수 (10Hz이므로 작게)
EMA_ALPHA_SLOW = 0.2
EMA_ALPHA_FAST = 0.6
DELTA_RAW_FAST = 250
PRINT_EVERY    = 0.5
LOOP_SLEEP     = 0.1    # 10Hz에 맞춰 루프 간격을 100ms로

# Zero-Lock
ZERO_LOCK_THRESHOLD = 5.0  # g
ZERO_LOCK_MIN_TIME  = 0.3  # s

# Sequence thresholds
TRIGGER_MIN   = 20000.0
TRIGGER_MAX   = 120000.0
BRANCH_SPLIT  = 65000.0
PAUSE_UNDER   = 20000.0
PAUSE_HOLD_S  = 3.0
SEQ1_HOLD_S   = 1.0

# -------------------------------
# Shared state / events
# -------------------------------
stop_event    = threading.Event()
zero_request  = threading.Event()
cal_request   = threading.Event()
state_lock    = threading.Lock()

zero_raw = None
cur_weight_g = 0.0
curA = A

def _install_sig_handlers():
    def _h(sig, frame): stop_event.set()
    signal.signal(signal.SIGINT, _h)
    signal.signal(signal.SIGTERM, _h)

# -------------------------------
# HX711 helpers
# -------------------------------
def _measure_zero_raw():
    return _read_ready_mean(ZERO_SAMPLES)

def _auto_zero(tag="start") -> bool:
    global zero_raw
    raw0 = _measure_zero_raw()
    if raw0 is None:
        print(f"[CAL] {tag}: zero measurement failed")
        return False
    with state_lock:
        zero_raw = raw0
    print(f"[CAL] {tag}: zero_raw set to {raw0}")
    return True

def _read_ready_mean(target_count=READ_SAMPLES, timeout_s=0.25):
    """HX711 is_ready()가 True일 때만 샘플을 모아 평균.
       -1, 0, None 같은 실패 샘플은 버림."""
    total, cnt = 0, 0
    deadline = time.time() + timeout_s
    while cnt < target_count and time.time() < deadline and not stop_event.is_set():
        # 준비됐을 때만 읽기
        if hasattr(hx, "is_ready") and not hx.is_ready():
            time.sleep(0.001)
            continue

        v = hx.get_raw_data_mean(1)  # 1샘플만 읽기
        if v is None or v in (-1, 0):
            time.sleep(0.001)
            continue

        total += int(v)
        cnt += 1

    return int(round(total / cnt)) if cnt > 0 else None

# -------------------------------
# Video helpers
# -------------------------------
def _fps_delay_ms(cap) -> int:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3: fps = 30.0
    d = max(1, int(round(1000.0 / fps)))
    return max(1, int(round(d / max(1e-6, SPEED_SCALE))))

def _show_frame(frame, delay_ms):
    if not HEADLESS:
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == 27: stop_event.set()
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
    cap, delay_ms = _open_video(path)
    if cap is None: return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    t_sec = 0.0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: break
        if on_tick: on_tick(t_sec, False)
        _show_frame(frame, delay_ms)
        t_sec += 1.0 / fps
    cap.release()
    return not stop_event.is_set()

def _play_pause_at(path: str, pause_time: float, until_cond):
    cap, delay_ms = _open_video(path)
    if cap is None: return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    t_sec, paused_frame = 0.0, None
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: 
            cap.release(); return not stop_event.is_set()
        if t_sec + 1e-6 >= pause_time and paused_frame is None:
            paused_frame = frame.copy(); break
        _show_frame(frame, delay_ms); t_sec += 1.0 / fps
    while not stop_event.is_set() and not until_cond():
        _show_frame(paused_frame, delay_ms)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: break
        _show_frame(frame, delay_ms)
    cap.release(); return not stop_event.is_set()

# -------------------------------
# HX711 reader thread
# -------------------------------
def hx711_reader():
    global zero_raw, cur_weight_g, curA, A
    hx.reset()
    if not _auto_zero("start"): zero_raw = 0

    smoothed_raw, zero_win_start, last_print = None, None, 0.0

    try:
        while not stop_event.is_set():
            if zero_request.is_set():
                time.sleep(0.5)
                if _auto_zero("re-zero"): smoothed_raw, zero_win_start = None, None
                zero_request.clear()

            raw = _read_ready_mean(target_count=READ_SAMPLES)
            if raw is not None:
                if smoothed_raw is None: smoothed_raw = raw
                else:
                    delta = abs(raw - smoothed_raw)
                    alpha = EMA_ALPHA_FAST if delta >= DELTA_RAW_FAST else EMA_ALPHA_SLOW
                    smoothed_raw = alpha * raw + (1.0 - alpha) * smoothed_raw

                with state_lock: zr, curA = zero_raw, A
                weight = curA * (smoothed_raw - zr)
                cur_weight_g = float(weight)

                now = time.time()
                if abs(weight) < ZERO_LOCK_THRESHOLD:
                    if zero_win_start is None: zero_win_start = now
                    elif (now - zero_win_start) >= ZERO_LOCK_MIN_TIME:
                        with state_lock: zero_raw = smoothed_raw
                        zero_win_start = None
                        print("[CAL] zero-lock: zero_raw snapped to EMA")
                else: zero_win_start = None

                if cal_request.is_set():
                    with state_lock: zr_local = zero_raw
                    delta_counts = smoothed_raw - zr_local
                    if abs(delta_counts) > 1.0:
                        newA = CAL_MASS_G / float(delta_counts)
                        with state_lock: oldA, A = A, newA
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
# Sequences
# -------------------------------
def get_weight(): return float(cur_weight_g)

def seq01_wait():
    print("[SEQ] 01(wait) start")
    while not stop_event.is_set():
        triggered, inrange_start = False, None
        def on_tick(t, is_last):
            nonlocal triggered, inrange_start
            w = get_weight()
            inrange = (TRIGGER_MIN <= w < TRIGGER_MAX)
            if inrange:
                if inrange_start is None: inrange_start = time.time()
                elif (time.time() - inrange_start) >= SEQ1_HOLD_S and not triggered:
                    print(f"[SEQ] 01: in-range ≥{SEQ1_HOLD_S}s (w≈{w:.0f} g)"); triggered = True
            else: inrange_start = None
        ok = _play_once("01.mp4", on_tick)
        if not ok: return False
        if triggered:
            print("[SEQ] 01 -> 02"); return True

def seq02_measure():
    print("[SEQ] 02(measure) start")
    start_time, valid_sum, valid_cnt, drop_detected = time.time(), 0.0, 0, False
    def tick(_t, _is_last):
        nonlocal valid_sum, valid_cnt, drop_detected
        w, elapsed = get_weight(), time.time() - start_time
        if elapsed >= 2.0:
            running_avg = (valid_sum / valid_cnt) if valid_cnt > 0 else w
            if valid_cnt > 0 and w <= running_avg * 0.5: drop_detected = True
            valid_sum += w; valid_cnt += 1
    for rep in range(2):
        ok = _play_once("02.mp4", tick)
        if not ok: return False, None
    avg = (valid_sum / valid_cnt) if valid_cnt else 0.0
    print(f"[SEQ] 02 avg={avg:.2f} g (cnt={valid_cnt}), drop_detected={drop_detected}")
    if drop_detected: return True, "01"
    if avg < TRIGGER_MIN: return True, "01"
    elif avg < BRANCH_SPLIT: return True, "03"
    else: return True, "03-1"

def seq03_pause_then_resume(video_name):
    print(f"[SEQ] {video_name} start (pause@2.0s)")
    under_start = [None]
    def cond():
        w, now = get_weight(), time.time()
        if w < PAUSE_UNDER:
            if under_start[0] is None: under_start[0] = now
            return (now - under_start[0]) >= PAUSE_HOLD_S
        else: under_start[0] = None; return False
    ok = _play_pause_at(video_name, pause_time=2.0, until_cond=cond)
    if not ok: return False
    print(f"[SEQ] {video_name} -> 01"); return True

def run_sequences():
    while not stop_event.is_set():
        if not seq01_wait(): break
        ok, branch = seq02_measure()
        if not ok: break
        if branch == "01": continue
        elif branch == "03":
            if not seq03_pause_then_resume("03.mp4"): break
        elif branch == "03-1":
            if not seq03_pause_then_resume("03-1.mp4"): break

# -------------------------------
# MAIN
# -------------------------------
def main():
    _install_sig_handlers()
    t = threading.Thread(target=hx711_reader, daemon=True); t.start()
    try: run_sequences()
    finally:
        stop_event.set()
        if not HEADLESS:
            try: cv2.destroyAllWindows()
            except Exception: pass
        t.join(timeout=2.0)
        GPIO.cleanup(); sys.exit(0)

if __name__ == "__main__": main()
