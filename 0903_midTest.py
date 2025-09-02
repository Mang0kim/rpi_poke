#!/usr/bin/env python3
import os, time, json, threading
import cv2
from HX711 import *

# -------------------- 설정 --------------------
CALIB_PATH = "hx711_calibration.json"
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VID_DIR    = os.path.join(BASE_DIR, "vid")
WINDOW     = "SmartScale"
ESC_KEY    = 27

# 측정 관련
THRESH_KG = 1.0
HOLD_TO_MEASURE_READY = 1.5   # seq1 -> seq2 (≥1kg 1.5s 유지)
HOLD_STABLE_BIN       = 3.0   # seq2 안정 판정 (3초 같은 kg bin)
FREEZE_AT_SEC         = 1.5   # seq3 freeze 시점
SAMPLES = 10                  # hx.weight(10) ≈ 1초
SPEED_SCALE = 1.0             # 항상 1.0배속 재생
# ---------------------------------------------

# 전역
weight_kg = 0.0
running = True

def load_calib():
    with open(CALIB_PATH, "r") as f:
        return json.load(f)

def weight_reader():
    """백그라운드에서 1초마다 무게 갱신"""
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
    cv2.setNumThreads(1)

def _open_cap(path):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[WARN] cannot open: {path}")
        return None
    return cap

def _play_video(path, until_frame=None, freeze=False, hold_cond=None, back_cond=None):
    """심플+정밀 타이밍 영상 재생"""
    cap = _open_cap(path)
    if cap is None: return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = (1.0 / fps) / SPEED_SCALE
    t_next = time.perf_counter()
    frame_idx = -1
    last = None
    esc = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        last = frame

        # 오버레이
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, h-50), (220, h-12), (0,0,0), -1)
        cv2.putText(frame, f"{weight_kg:.2f} kg", (18, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW, frame)

        t_next += delay
        remain = t_next - time.perf_counter()
        k = cv2.waitKey(int(remain*1000) if remain>0 else 1) & 0xFF
        if k == ESC_KEY:
            esc = True
            break

        if until_frame is not None and frame_idx >= until_frame:
            break

    cap.release()

    # freeze 모드
    if freeze and last is not None:
        while hold_cond():
            f2 = last.copy()
            h,w = f2.shape[:2]
            cv2.rectangle(f2, (10, h-50), (220, h-12), (0,0,0), -1)
            cv2.putText(f2, f"{weight_kg:.2f} kg", (18, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, f2)
            k = cv2.waitKey(50) & 0xFF
            if k == ESC_KEY:
                esc = True
                break
            if back_cond and back_cond():
                break
    return esc

def kg_bin_round(x):
    if x < 1.0: return None
    b = int(round(x))
    return max(1, min(100, b))

def main():
    global running
    ensure_fullscreen()

    t = threading.Thread(target=weight_reader, daemon=True)
    t.start()

    state = 1
    seq1_hold_start = None
    seq2_bin = None
    seq2_bin_start = None

    try:
        while True:
            # --- SEQ1 ---
            while state == 1:
                if weight_kg >= THRESH_KG:
                    if seq1_hold_start is None:
                        seq1_hold_start = time.time()
                    elif time.time() - seq1_hold_start >= HOLD_TO_MEASURE_READY:
                        esc = _play_video(os.path.join(VID_DIR,"01_fix.mp4"))
                        if esc: raise KeyboardInterrupt
                        state = 2
                        seq1_hold_start = None
                        seq2_bin = None
                        seq2_bin_start = None
                        break
                else:
                    seq1_hold_start = None
                esc = _play_video(os.path.join(VID_DIR,"01_fix.mp4"))
                if esc: raise KeyboardInterrupt

            # --- SEQ2 ---
            while state == 2:
                current_bin = kg_bin_round(weight_kg)
                now = time.time()

                if weight_kg < THRESH_KG:
                    esc = _play_video(os.path.join(VID_DIR,"02_fix.mp4"))
                    if esc: raise KeyboardInterrupt
                    state = 1
                    break

                if current_bin is None:
                    seq2_bin = None; seq2_bin_start=None
                elif seq2_bin is None or current_bin != seq2_bin:
                    seq2_bin = current_bin; seq2_bin_start=now
                elif now - seq2_bin_start >= HOLD_STABLE_BIN:
                    esc = _play_video(os.path.join(VID_DIR,"02_fix.mp4"))
                    if esc: raise KeyboardInterrupt
                    result_bin = seq2_bin
                    state = 3
                    break

                esc = _play_video(os.path.join(VID_DIR,"02_fix.mp4"))
                if esc: raise KeyboardInterrupt

            # --- SEQ3 ---
            while state == 3:
                filename = f"ScaleCustom_txt_{result_bin:02d}_fix.mp4"
                path = os.path.join(VID_DIR,"txt",filename)
                cap = _open_cap(path)
                if cap is None: state=1; break
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                until_frame = int(round(fps*FREEZE_AT_SEC))
                cap.release()
                esc = _play_video(path, until_frame, freeze=True,
                                  hold_cond=lambda: weight_kg>=THRESH_KG,
                                  back_cond=lambda: weight_kg<THRESH_KG)
                if esc: raise KeyboardInterrupt
                if weight_kg < THRESH_KG:
                    state = 1; break
                time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        running=False
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
