#!/usr/bin/env python3
import time, json
from HX711 import *

CALIB_PATH = "hx711_calibration.json"

def main():
    # JSON에서 보정값 불러오기
    with open(CALIB_PATH, "r") as f:
        cfg = json.load(f)

    with SimpleHX711(cfg["data_pin"], cfg["clk_pin"], 1, 0) as hx:
        hx.setUnit(Mass.Unit.KG)

        a, b = cfg["a"], cfg["b"]  # 선형 보정 계수

        while True:
            measured = float(hx.weight(35))   # 샘플 35개 평균
            corrected = a * measured + b
            print(f"{corrected:.3f} kg")
            time.sleep(1)   # 1초 간격 출력

if __name__ == "__main__":
    main()
