#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HX711 + OpenCV Sequencer
- Adaptive EMA + Zero-Lock + Z/C controls
- Sequences:
  01(wait >=1kg for 2s, pre-assign result) -> 02(measure twice) -> 03/03-1(pause/resume) -> 01
"""

import cv2, time, threading, signal, sys
from hx711 import HX711
import RPi.GPIO as GPIO
GPIO.setwarnings(False)

# -------------------------------
# USER SETTINGS
# -------------------------------
WINDOW_NAME   = "Player"
HEADLESS      = False
SPEED_SCALE   = 1.0

# Calibration slope (press 'C' to recalibrate with known mass)
A = 0.03883
CAL_MASS_G = 69200.0

# Sampling / smoothing
ZERO_SAMPLES   = 15
READ_SAMPLES   = 2
EMA_ALPHA_SLOW = 0.12
EMA_ALPHA_FAST = 0.50
DELTA_RAW_FAST = 900
PRINT_EVERY    = 0.2
LOOP_SLEEP     = 0.02

# Zero-Lock
ZERO_LOCK_THRESHOLD = 100.0  # g
ZERO_LOCK_MIN_TIME  = 0.7    # s

# Sequence thresholds (기본값 유지; 논리상 핵심은 1kg/2s)
TRIGGER_MIN  = 1000.0
TRIGGER_MAX  = 100000.0
BRANCH_SPLIT = 65000.0
PAUSE_UNDER  = 1000.0
PAUSE_HOLD_S = 2.0

# --- Guard statistics.stdev globally (prevents "two data points" crash) ---
import statistics as _stats
_orig_stdev = _stats.stdev
def _safe_stdev(vals):
    vals = list(vals)
    if len(vals) >= 2:
        return _orig_stdev(vals)
    return 0.0
_stats.stdev = _safe_stdev

# --- Read only when ready; drop invalid samples; average outside the lib filter ---
def _read_ready_mean(target_count, timeout_s=0.5):
    total, cnt = 0, 0
    deadline = time.time() + timeout_s
    while cnt < target_count and time.time() < deadline and not stop_event.is_set():
        if hasattr(hx, "is_ready") and not hx.is_ready():
            time.sleep(0.001); continue
        v = hx.get_raw_data_mean(1)
        if v in (None, 0, -1):
            time.sleep(0.001); continue
        if abs(v) > 5_000_000:  # 비정상 스파이크 버림
            continue
        total += int(v); cnt += 1
    return int(round(total / cnt)) if cnt > 0 else None

# --- Running mean / stdev (Welford) ---
class RunningStats:
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0
    def push(self, x: float):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)
    @property
    def count(self): return self.n
    @property
    def avg(self):   return self.mean if self.n > 0 else 0.0
    @property
    def stdev(self): return (self.M2/(self.n-1))**0.5 if self.n >= 2 else 0.0

# -------------------------------
# HX711 / GPIO
# -------------------------------
GPIO.setmode(GPIO.BCM)
hx = HX711(dout_pin=5, pd_sck_pin=6)  # DOUT=5, SCK=6

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
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
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
    off = _read_ready_mean(20)
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

            raw = _read_ready_mean(READ_SAMPLES)
            if raw is not None:
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
# Helpers for sequence decisions
# -------------------------------
def get_weight():
    return float(cur_weight_g)

def _result_code_from_weight(w: float):
    """1kg 단위 bin 홀/짝으로 03 / 03-1 결정"""
    if w < 1000.0:
        return None
    bin_idx = int(w // 1000.0)  # 1000~1999 -> 1, 2000~2999 -> 2, ...
    return "03" if (bin_idx % 2 == 1) else "03-1"

# -------------------------------
# Sequence logic
# -------------------------------
def seq01_wait():
    """
    01.mp4 반복 재생.
    조건: 1000g 이상이 2초 연속 유지되면 시퀀스2 진입.
    이때의 무게로 결과 영상을 미리 배정(1kg bin 홀/짝).
    """
    print("[SEQ] 01(wait) start")
    trigger_armed = False
    assigned_code = None
    hold_start = None

    while not stop_event.is_set():
        def on_tick(t, _is_last):
            nonlocal trigger_armed, assigned_code, hold_start
            w = get_weight()
            if w >= 1000.0:
                if hold_start is None:
                    hold_start = time.time()
                elif not trigger_armed and (time.time() - hold_start) >= 2.0:
                    # 2초 유지 충족 → 바로 배정
                    assigned_code = _result_code_from_weight(w)
                    trigger_armed = True
            else:
                hold_start = None  # 조건 리셋

        ok = _play_once("01.mp4", on_tick)
        if not ok: return False, None
        if trigger_armed:
            print(f"[SEQ] 01 -> 02 (assigned result={assigned_code})")
            return True, assigned_code

def seq02_measure(preassigned: str):
    """
    02.mp4 두 번 재생(항상 끝까지).
    시작 후 2초 버리고 러닝평균/표준편차 계산.
    - 드롭(현재 ≤ 러닝평균의 50%) 감지 시: 01로 복귀
    - 드롭 없으면: 01에서 미리 배정한 결과 영상을 사용
    """
    print("[SEQ] 02(measure) start")
    start_time = time.time()
    stats = RunningStats()
    drop_detected = False

    def tick(_t, _is_last):
        nonlocal drop_detected, stats
        w = get_weight()
        elapsed = time.time() - start_time
        if elapsed >= 2.0:
            stats.push(w)
            if stats.count >= 2 and w <= stats.avg * 0.5:
                drop_detected = True

    for _ in range(2):
        ok = _play_once("02.mp4", tick)
        if not ok: return False, None

    avg = stats.avg
    sd  = stats.stdev
    print(f"[SEQ] 02 avg={avg:.2f} g, stdev={sd:.2f} g (cnt={stats.count}), drop_detected={drop_detected}")

    if drop_detected or avg < TRIGGER_MIN:
        print("[SEQ] 02 -> 01 (drop or avg below min)")
        return True, "01"

    # 드롭이 없으면 시퀀스1에서 배정한 결과 사용
    branch = preassigned if preassigned in ("03", "03-1") else _result_code_from_weight(avg) or "03"
    print(f"[SEQ] 02 -> {branch} (preassigned)")
    return True, branch

def seq03_pause_then_resume(video_name):
    """
    03/03-1: 2.0초에서 일시정지.
    weight<PAUSE_UNDER 가 PAUSE_HOLD_S 이상 유지되면 재생 재개 후 종료.
    """
    print(f"[SEQ] {video_name} start (pause@2.0s)")
    under_start = [None]
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
        ok, preassigned = seq01_wait()
        if not ok: break
        ok, branch = seq02_measure(preassigned)
        if not ok: break
        if branch == "01":
            continue
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
