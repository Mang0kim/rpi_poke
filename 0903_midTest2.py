#!/usr/bin/env python3
import os, time, json, threading, subprocess
from HX711 import *

# -------------------- Paths & Const --------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VID_DIR    = os.path.join(BASE_DIR, "vid")
CALIB_PATH = os.path.join(BASE_DIR, "hx711_calibration.json")

THRESH_KG = 1.0
SEQ1_HOLD_SEC = 1.5      # Seq1 -> Seq2: >=1kg 1.5초 유지
SEQ2_STABLE_SEC = 3.0    # Seq2 안정: round(kg) 3초 동일
SEQ3_FREEZE_SEC = 1.5    # Seq3: 1.5초까지 재생 후 pause
SAMPLES = 10             # hx.weight(10) ≈ 1초
MPV_SPEED = "1.0"        # 항상 1.0배속

# -------------------- Globals --------------------
weight_kg = 0.0          # 보정 a,b 적용 후 실시간 무게
running = True
tare_offset = 0.0        # Seq1에서 영점 업데이트
tare_window_s = 1.0      # 영점 평균 윈도 (초)

# -------------------- Helpers --------------------
def load_calib():
    with open(CALIB_PATH, "r") as f:
        return json.load(f)

def weight_reader():
    """1초 주기로 무게 갱신 (a,b 보정 적용)."""
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

def effective_weight():
    """영점 반영된 무게 (음수 방지)."""
    return max(0.0, weight_kg - tare_offset)

def get_video_path(stem, subdir=None):
    """
    파일 탐색 유틸: 우선 *_fix.mp4 찾고, 없으면 .mp4 사용.
    stem: '01', '02', 'ScaleCustom_txt_01' 등
    """
    base = os.path.join(VID_DIR, subdir) if subdir else VID_DIR
    cand_fix = os.path.join(base, f"{stem}_fix.mp4")
    cand     = os.path.join(base, f"{stem}.mp4")
    if os.path.isfile(cand_fix): return cand_fix
    return cand

def mpv_play(path, freeze_sec=None):
    """
    mpv로 영상 재생.
    - freeze_sec=None: 끝까지 재생
    - freeze_sec=t: t초까지 재생 후 pause (그 프레임에서 정지)
    """
    if freeze_sec is None:
        cmd = ["mpv", "--fs", "--no-osc", "--really-quiet",
               f"--speed={MPV_SPEED}", path]
    else:
        # t초까지만 재생 후 정지
        cmd = ["mpv", "--fs", "--no-osc", "--really-quiet",
               f"--speed={MPV_SPEED}", f"--end={freeze_sec}", path, "--pause"]
    subprocess.run(cmd)

def kg_bin_round(x):
    """안정 판정용: round(kg). 1~100만 유효, <1이면 None."""
    if x < 1.0:
        return None
    b = int(round(x))
    return max(1, min(100, b))

def kg_bin_floor_for_result(x):
    """결과 선택용: [N, N+1) -> N (1~100 클램프)."""
    if x < 1.0:
        return None
    b = int(x)  # floor
    if b < 1: b = 1
    if b > 100: b = 100
    return b

def seq1_update_tare(start_time):
    """Seq1 동안 주기적으로 영점(tare_offset) 업데이트 (평균 1초)."""
    global tare_offset
    # 1초 윈도로 평균 잡기
    acc = 0.0; cnt = 0
    t_begin = time.time()
    while time.time() - t_begin < tare_window_s:
        acc += weight_kg
        cnt += 1
        time.sleep(0.05)
    if cnt > 0:
        tare_offset = acc / cnt

# -------------------- Main FSM --------------------
def main():
    global running, tare_offset

    # sanity
    if not os.path.isdir(VID_DIR):
        print("[ERR] Video folder not found:", VID_DIR)
        return

    # start weight thread
    t = threading.Thread(target=weight_reader, daemon=True)
    t.start()

    state = 1
    seq1_hold_start = None
    seq2_bin = None
    seq2_bin_start = None

    try:
        while True:
            # ---------------- SEQ 1: 대기 (<=vid/01) ----------------
            while state == 1:
                # 영점 반영(현재 상태)
                seq1_update_tare(time.time())

                eff = effective_weight()
                if eff >= THRESH_KG:
                    if seq1_hold_start is None:
                        seq1_hold_start = time.time()
                    elif time.time() - seq1_hold_start >= SEQ1_HOLD_SEC:
                        # 조건 충족 → 하지만 현재 영상 1회 끝까지 재생 후 전환
                        mpv_play(get_video_path("01"))
                        state = 2
                        seq1_hold_start = None
                        seq2_bin = None
                        seq2_bin_start = None
                        break
                else:
                    seq1_hold_start = None

                # 항상 끝까지
                mpv_play(get_video_path("01"))

            # ---------------- SEQ 2: 측정 (<=vid/02) ----------------
            while state == 2:
                eff = effective_weight()
                now = time.time()

                # 아래로 떨어지면, 현재 영상 끝난 후 Seq1 복귀
                if eff < THRESH_KG:
                    mpv_play(get_video_path("02"))
                    state = 1
                    break

                current_bin = kg_bin_round(eff)
                if current_bin is None:
                    seq2_bin = None; seq2_bin_start = None
                elif (seq2_bin is None) or (current_bin != seq2_bin):
                    seq2_bin = current_bin
                    seq2_bin_start = now
                elif now - seq2_bin_start >= SEQ2_STABLE_SEC:
                    # 조건 충족 → 현재 영상 끝까지 재생 후 결과로
                    mpv_play(get_video_path("02"))
                    result_bin = kg_bin_floor_for_result(eff)  # [N,N+1) 매핑
                    if result_bin is None:
                        state = 1
                    else:
                        state = 3
                    break

                # 항상 끝까지
                mpv_play(get_video_path("02"))

            # ---------------- SEQ 3: 결과 (<=vid/txt/ScaleCustom_txt_XX) ----------------
            while state == 3:
                stem = f"ScaleCustom_txt_{result_bin:02d}"
                path = get_video_path(stem, subdir="txt")

                # 1) 1.5초 지점까지 재생
                mpv_play(path, freeze_sec=SEQ3_FREEZE_SEC)

                # 2) 하중 유지되면 그 프레임에서 '일시정지 유지'
                #    (무게 < 1kg 되면 Seq1로)
                while effective_weight() >= THRESH_KG:
                    time.sleep(0.1)

                state = 1
                break

    except KeyboardInterrupt:
        pass
    finally:
        running = False

if __name__ == "__main__":
    main()
