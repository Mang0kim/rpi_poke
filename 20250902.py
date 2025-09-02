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
ZERO_SAMPLES  = 30
READ_SAMPLES  = 12
EMA_ALPHA_SLOW = 0.12
EMA_ALPHA_FAST = 0.50
DELTA_RAW_FAST = 900
PRINT_EVERY    = 1.0
LOOP_SLEEP     = 1

# Zero-Lock
ZERO_LOCK_THRESHOLD = 100.0  # g
ZERO_LOCK_MIN_TIME  = 0.7  # s

# Sequence thresholds
TRIGGER_MIN  = 1000.0      # >=
TRIGGER_MAX  = 100000.0     # <
BRANCH_SPLIT = 65000.0      # seq02 avg split
PAUSE_UNDER  = 1000.0      # < for 2 seconds
PAUSE_HOLD_S = 2.0

# --- Guard statistics.stdev globally (prevents "two data points" crash) ---
import statistics as _stats

_orig_stdev = _stats.stdev
def _safe_stdev(vals):
    vals = list(vals)
    if len(vals) >= 2:
        return _orig_stdev(vals)   # 표본 표준편차
    return 0.0                     # 표본이 0~1개면 0.0으로 처리

_stats.stdev = _safe_stdev

# --- Read only when ready; drop invalid samples; average outside the lib filter ---
def _read_ready_mean(target_count, timeout_s=0.5):
    """
    HX711 is_ready()일 때만 1샘플씩 읽어 평균.
    -1/0/None 같은 실패 샘플은 버림.
    target_count개 모으거나 timeout 지나면 종료.
    """
    total, cnt = 0, 0
    deadline = time.time() + timeout_s
    while cnt < target_count and time.time() < deadline and not stop_event.is_set():
        if hasattr(hx, "is_ready") and not hx.is_ready():
            time.sleep(0.001); continue

        # 라이브러리 필터 경유를 피하려고 "한 번에 1개"만 요청
        v = hx.get_raw_data_mean(1)
        if v in (None, 0, -1):
            time.sleep(0.001)
            continue
        if abs(v) > 5000000:   # 비정상 스파이크 버림
            continue

        total += int(v); cnt += 1

    return int(round(total / cnt)) if cnt > 0 else None


# --- Running mean / stdev (Welford) ---
class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # sum of squares of differences from the current mean

    def push(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    @property
    def count(self) -> int:
        return self.n

    @property
    def avg(self) -> float:
        return self.mean if self.n > 0 else 0.0

    @property
    def stdev(self) -> float:
        # 표본 표준편차 (샘플 2개 미만이면 0.0 반환 → 예외 없음)
        if self.n >= 2:
            return (self.M2 / (self.n - 1)) ** 0.5
        return 0.0


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
    """02.mp4 두 번 재생(항상 끝까지).
       시작 후 2초는 버리고, 이후 측정으로 러닝 평균/표준편차 계산.
       drop 규칙: 현재값 <= 러닝평균의 50% 이하면 플래그.
       두 번 재생을 모두 마친 뒤 drop_detected면 01로 복귀."""
    print("[SEQ] 02(measure) start")

    start_time = time.time()
    stats = RunningStats()    # ← 러닝 통계 시작
    drop_detected = False

    def tick(_t, _is_last):
        nonlocal drop_detected, stats
        w = get_weight()
        elapsed = time.time() - start_time

        # 초기 2초 버림
        if elapsed >= 2.0:
            # 러닝 평균/표준편차 업데이트 (예외 없는 안전 계산)
            stats.push(w)

            # 현재값이 러닝 평균의 50% 이하이면 드롭 감지
            if stats.count >= 2 and w <= stats.avg * 0.5:
                drop_detected = True

    # 02.mp4 두 번 끝까지 재생
    for _ in range(2):
        ok = _play_once("02.mp4", tick)
        if not ok:
            return False, None

    avg = stats.avg
    sd  = stats.stdev  # 필요한 경우 로그/판정에 사용 가능

    print(f"[SEQ] 02 avg={avg:.2f} g, stdev={sd:.2f} g (cnt={stats.count}), drop_detected={drop_detected}")

    # 새 규칙 최우선
    if drop_detected:
        print("[SEQ] 02 -> 01 (detected current <= 50% of running avg)")
        return True, "01"

    # 기존 분기
    if avg < TRIGGER_MIN:
        print("[SEQ] 02 -> 01 (avg < trigger min)")
        return True, "01"
    elif avg < BRANCH_SPLIT:
        print("[SEQ] 02 -> 03 (avg < split)")
        return True, "03"
    else:
        print("[SEQ] 02 -> 03-1 (avg >= split)")
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
