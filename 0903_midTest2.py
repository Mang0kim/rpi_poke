#!/usr/bin/env python3
import os, time, json, threading, subprocess
from HX711 import *

# -------------------- 설정 --------------------
CALIB_PATH = "hx711_calibration.json"
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VID_DIR    = os.path.join(BASE_DIR, "vid")

THRESH_KG = 1.0
HOLD_TO_MEASURE_READY = 1.5   # seq1 -> seq2 (≥1kg 1.5s 유지)
HOLD_STABLE_BIN       = 3.0   # seq2 안정 판정 (3초 같은 kg bin)
FREEZE_AT_SEC         = 1.5   # seq3 freeze 시점
SAMPLES = 10                  # hx.weight(10) ≈ 1초
SPEED_SCALE = "1.0"           # mpv 재생속도 (문자열)

weight_kg = 0.0
running = True
# ---------------------------------------------

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

def play_mpv(path, freeze_sec=None):
    """
    mpv 실행 (풀스크린, UI 없음, 지정 속도)
    freeze_sec 지정 시: 해당 시점까지 재생 후 멈춤 (--pause --start).
    """
    if freeze_sec is None:
        cmd = ["mpv", "--fs", "--no-osc", "--really-quiet",
               f"--speed={SPEED_SCALE}", path]
    else:
        cmd = ["mpv", "--fs", "--no-osc", "--really-quiet",
               f"--speed={SPEED_SCALE}", f"--end={freeze_sec}", path,
               "--pause"]  # freeze 유지
    subprocess.run(cmd)

def kg_bin_round(x):
    if x < 1.0: return None
    b = int(round(x))
    return max(1, min(100, b))

def main():
    global running
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
                        play_mpv(os.path.join(VID_DIR,"01_fix.mp4"))
                        state = 2
                        seq1_hold_start = None
                        seq2_bin = None
                        seq2_bin_start = None
                        break
                else:
                    seq1_hold_start = None
                play_mpv(os.path.join(VID_DIR,"01_fix.mp4"))

            # --- SEQ2 ---
            while state == 2:
                current_bin = kg_bin_round(weight_kg)
                now = time.time()

                if weight_kg < THRESH_KG:
                    play_mpv(os.path.join(VID_DIR,"02_fix.mp4"))
                    state = 1
                    break

                if current_bin is None:
                    seq2_bin = None; seq2_bin_start=None
                elif seq2_bin is None or current_bin != seq2_bin:
                    seq2_bin = current_bin; seq2_bin_start=now
                elif now - seq2_bin_start >= HOLD_STABLE_BIN:
                    play_mpv(os.path.join(VID_DIR,"02_fix.mp4"))
                    result_bin = seq2_bin
                    state = 3
                    break

                play_mpv(os.path.join(VID_DIR,"02_fix.mp4"))

            # --- SEQ3 ---
            while state == 3:
                filename = f"ScaleCustom_txt_{result_bin:02d}_fix.mp4"
                path = os.path.join(VID_DIR,"txt",filename)
                # 1.5초까지 재생하고 멈춰서 freeze 유지
                play_mpv(path, freeze_sec=FREEZE_AT_SEC)
                # freeze 상태 유지: 무게가 <1kg 되면 종료
                while weight_kg >= THRESH_KG:
                    time.sleep(0.1)
                state = 1
                break

    except KeyboardInterrupt:
        pass
    finally:
        running=False

if __name__=="__main__":
    main()
