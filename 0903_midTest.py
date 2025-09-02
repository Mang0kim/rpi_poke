#!/usr/bin/env python3
# smart_scale_fsm.py
# Requirements:
# - /vid/01.mp4, /vid/02.mp4
# - /vid/txt/ScaleCustom_txt_01.mp4 ... _100.mp4
# - hx711_calibration.json with {"data_pin":5,"clk_pin":6,"a":X,"b":Y}

import os, time, json, threading
import cv2
from HX711 import *

CALIB_PATH = "hx711_calibration.json"
VID_DIR    = "/vid"
WINDOW     = "SmartScale"
ESC_KEY    = 27

# thresholds & timings (seconds)
THRESH_KG = 1.0
HOLD_TO_MEASURE_READY = 1.5   # seq1 -> seq2 조건 (>=1kg 연속 유지)
HOLD_STABLE_BIN = 3.0         # seq2에서 같은 kg단위 연속 유지 시간
FREEZE_AT_SEC = 1.5           # seq3에서 일시정지 시점
SAMPLES = 10                  # hx.weight(10) ≈ 1초 @ 10SPS

# globals
weight_kg = 0.0
running = True

def load_calib():
    with open(CALIB_PATH, "r") as f:
        return json.load(f)

def weight_reader():
    """Reads weight every ~1s and updates global weight_kg using a,b correction."""
    global weight_kg, running
    cfg = load_calib()
    a, b = cfg["a"], cfg["b"]
    with SimpleHX711(cfg["data_pin"], cfg["clk_pin"], 1, 0) as hx:
        hx.setUnit(Mass.Unit.KG)
        while running:
            t0 = time.time()
            try:
                measured = float(hx.weight(SAMPLES))
                weight_kg = a * measured + b
            except Exception:
                pass
            dt = time.time() - t0
            time.sleep(max(0, 1.0 - dt))

def ensure_fullscreen():
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def calc_delay_ms(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0: fps = 30.0
    return int(round(1000.0 / fps)), fps

def play_to_end_loop(path, condition_check=None):
    """
    Play video to its end (always). If condition_check() returns True, we still
    finish the current video, then return.
    Returns True if ESC was pressed (to exit program).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[WARN] cannot open: {path}")
        return False
    delay, _ = calc_delay_ms(cap)
    esc = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # draw overlay weight (optional; comment out if not needed)
        f = frame
        h, w = f.shape[:2]
        cv2.rectangle(f, (10, h-50), (220, h-12), (0,0,0), -1)
        cv2.putText(f, f"{weight_kg:.2f} kg", (18, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW, f)
        k = cv2.waitKey(delay) & 0xFF
        if k == ESC_KEY:
            esc = True
            break
        # we *don't* break early; we let it finish to last frame
    cap.release()

    # after finishing, check condition to move on
    if condition_check and condition_check():
        pass
    return esc

def play_until_and_freeze(path, freeze_time_sec, hold_condition, fallback_condition):
    """
    Play video until freeze_time_sec, then freeze that frame (show repeatedly).
    keep frozen while hold_condition() is True. If fallback_condition() becomes True,
    return to caller (to change state).
    Returns 'esc' pressed boolean.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[WARN] cannot open: {path}")
        return False
    delay, fps = calc_delay_ms(cap)
    target_frame = int(round(fps * freeze_time_sec))

    esc = False
    current_idx = -1
    last_frame = None

    while True:
        ok, frame = cap.read()
        if not ok:
            # shorter than freeze point → freeze last frame
            break
        current_idx += 1
        last_frame = frame
        f = frame
        h, w = f.shape[:2]
        cv2.rectangle(f, (10, h-50), (220, h-12), (0,0,0), -1)
        cv2.putText(f, f"{weight_kg:.2f} kg", (18, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW, f)
        k = cv2.waitKey(delay) & 0xFF
        if k == ESC_KEY:
            esc = True
            cap.release()
            return esc
        if current_idx >= target_frame:
            break

    cap.release()
    # Freeze on last_frame
    if last_frame is None:
        return esc
    while hold_condition():
        f2 = last_frame.copy()
        h, w = f2.shape[:2]
        cv2.rectangle(f2, (10, h-50), (220, h-12), (0,0,0), -1)
        cv2.putText(f2, f"{weight_kg:.2f} kg", (18, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, f2)
        k = cv2.waitKey(50) & 0xFF
        if k == ESC_KEY:
            return True
        if fallback_condition():
            break
    return esc

def kg_bin_floor_1_to_100(x):
    """Map weight to [1..100] bin: [1,2) -> 1, ... [100,101) -> 100; return None if <1."""
    if x < 1.0:
        return None
    b = int(x)  # floor
    if b < 1: b = 1
    if b > 100: b = 100
    return b

def kg_bin_round_1_to_100(x):
    """Alternative: round to nearest kg; more robust near boundaries."""
    if x < 1.0:
        return None
    b = int(round(x))
    if b < 1: b = 1
    if b > 100: b = 100
    return b

def main():
    global running
    ensure_fullscreen()

    # weight thread
    t = threading.Thread(target=weight_reader, daemon=True)
    t.start()

    # --- State machine ---
    state = 1
    seq1_hold_start = None
    seq2_bin = None
    seq2_bin_start = None

    try:
        while True:
            # -------- SEQ 1: Waiting (<1kg), play 01.mp4 looping --------
            while state == 1:
                # tare at current state (영점 반영)
                # NOTE: 우리의 보정은 a,b 기반이라 zero 명령은 생략(측정 기준은 a,b에 의해 정해짐).
                # "현재 상태로 영점 반영"을 강하게 원하면 a,b 대신 offset 보정 루틴을 별도로 추가하세요.

                # transition condition tracking
                if weight_kg >= THRESH_KG:
                    if seq1_hold_start is None:
                        seq1_hold_start = time.time()
                    elif time.time() - seq1_hold_start >= HOLD_TO_MEASURE_READY:
                        # condition met AFTER finishing this video
                        def cond(): return True
                        esc = play_to_end_loop(os.path.join(VID_DIR, "01.mp4"), cond)
                        if esc: raise KeyboardInterrupt
                        state = 2
                        seq1_hold_start = None
                        seq2_bin = None
                        seq2_bin_start = None
                        break
                else:
                    seq1_hold_start = None

                esc = play_to_end_loop(os.path.join(VID_DIR, "01.mp4"))
                if esc: raise KeyboardInterrupt

            # -------- SEQ 2: Measuring (>=1kg), play 02.mp4 looping --------
            while state == 2:
                # If drops below threshold, go back to seq1 after current video ends
                current_bin = kg_bin_round_1_to_100(weight_kg)  # use rounded bin
                now = time.time()

                if weight_kg < THRESH_KG:
                    def cond(): return True
                    esc = play_to_end_loop(os.path.join(VID_DIR, "02.mp4"), cond)
                    if esc: raise KeyboardInterrupt
                    state = 1
                    seq1_hold_start = None
                    break

                # Stability logic: bin unchanged for HOLD_STABLE_BIN seconds
                if current_bin is None:
                    seq2_bin = None
                    seq2_bin_start = None
                elif seq2_bin is None or current_bin != seq2_bin:
                    seq2_bin = current_bin
                    seq2_bin_start = now
                elif now - seq2_bin_start >= HOLD_STABLE_BIN:
                    # lock & transition after finishing this loop
                    locked_bin = seq2_bin
                    def cond(): return True
                    esc = play_to_end_loop(os.path.join(VID_DIR, "02.mp4"), cond)
                    if esc: raise KeyboardInterrupt
                    # pass value to seq3
                    state = 3
                    result_bin = locked_bin
                    break

                esc = play_to_end_loop(os.path.join(VID_DIR, "02.mp4"))
                if esc: raise KeyboardInterrupt

            # -------- SEQ 3: Result (freeze at 1.5s) --------
            while state == 3:
                # choose video by result_bin
                # 1kg 이상 2kg 미만 -> _01.mp4 ... 100kg 이상 101kg 미만 -> _100.mp4
                filename = f"ScaleCustom_txt_{result_bin:02d}.mp4"
                path = os.path.join(VID_DIR, "txt", filename)

                def hold_cond():
                    return weight_kg >= THRESH_KG

                def back_to_seq1_cond():
                    return weight_kg < THRESH_KG

                esc = play_until_and_freeze(path, FREEZE_AT_SEC, hold_cond, back_to_seq1_cond)
                if esc: raise KeyboardInterrupt

                if weight_kg < THRESH_KG:
                    state = 1
                    seq1_hold_start = None
                    break
                else:
                    # still holding ≥1kg: keep frozen loop (re-freeze the same frame)
                    # slight idle wait before re-entering freeze
                    time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
