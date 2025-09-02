#!/usr/bin/env python3
import time, json
from HX711 import *

CALIB_PATH = "hx711_calibration.json"
SAMPLES = 10  # ≈1초 목표 (10 SPS 기준)

def main():
    with open(CALIB_PATH, "r") as f:
        cfg = json.load(f)

    a, b = cfg["a"], cfg["b"]

    with SimpleHX711(cfg["data_pin"], cfg["clk_pin"], 1, 0) as hx:
        hx.setUnit(Mass.Unit.KG)

        while True:
            t0 = time.time()
            measured = float(hx.weight(SAMPLES))   # 더 적은 샘플 → 더 빠름
            corrected = a * measured + b
            print(f"{corrected:.3f} kg")

            # 루프를 정확히 1초 주기로 맞춤
            elapsed = time.time() - t0
            time.sleep(max(0, 1.0 - elapsed))

if __name__ == "__main__":
    main()
