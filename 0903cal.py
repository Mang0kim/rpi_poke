#!/usr/bin/env python3
# hx711_calibrate.py
# 1) 영점(offset) 측정 → 2) 기준추 올려 스케일(scale) 계산 → 3) JSON 저장

import time, json, statistics as stats
from hx711 import HX711   # endail/hx711 바인딩 모듈 예시 (맞는 모듈명 확인 필요)

CALIB_PATH = "hx711_calibration.json"

# -------------------
# Helper functions
# -------------------
def mean_raw(hx, n=20, delay_s=0.05):
    vals = []
    for _ in range(n):
        vals.append(hx.get_raw_data_mean(readings=1))  # raw 24bit count
        time.sleep(delay_s)
    return stats.mean(vals)

def calibrate(hx, known_mass_kg, n=30):
    print("Step 1: Make sure the scale is empty.")
    input("Press Enter when ready...")
    offset = mean_raw(hx, n=n)
    print(f"Offset(raw) = {offset}")

    print(f"Step 2: Place {known_mass_kg} kg weight.")
    input("Press Enter when ready...")
    raw_with_weight = mean_raw(hx, n=n)
    print(f"Raw with weight = {raw_with_weight}")

    delta = raw_with_weight - offset
    if abs(delta) < 1:
        raise RuntimeError("Calibration failed: no difference detected.")

    scale = delta / known_mass_kg  # raw counts per kg
    cfg = {"offset": offset, "scale": scale}
    with open(CALIB_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print("Calibration saved:", cfg)

if __name__ == "__main__":
    # HX711 연결 핀 번호에 맞게 수정
    hx = HX711(dout=5, pd_sck=6)  # 예시: GPIO5=DT, GPIO6=SCK
    hx.reset()

    calibrate(hx, known_mass_kg=1.000)  # 1kg 기준추 사용
