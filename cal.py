#!/usr/bin/env python3
# multi_point_calibrate.py
# 0kg / 1.75kg / 70kg 세 점으로 캘리브레이션

import time, json, statistics as stats
from HX711 import *   # endail/hx711 Python 바인딩

CALIB_PATH = "hx711_calibration.json"

DATA_PIN = 5
CLK_PIN  = 6

def mean_raw(hx, samples=35, repeats=10, delay=0.05):
    vals = []
    for _ in range(repeats):
        v = hx.rawDataMean(samples)   # SimpleHX711는 raw 리턴 메서드도 제공
        vals.append(int(v))
        time.sleep(delay)
    return stats.mean(vals)

def main():
    with SimpleHX711(DATA_PIN, CLK_PIN, 1, 0) as hx:
        hx.setUnit(Mass.Unit.KG)

        ref_points = []
        for known_mass in [0.0, 1.75, 70.0]:
            print(f"Place {known_mass} kg weight.")
            input("Press Enter when ready...")
            raw_val = mean_raw(hx)
            print(f"Raw at {known_mass} kg = {raw_val}")
            ref_points.append((raw_val, known_mass))

        # 최소자승법 (2점 이상 → 직선 피팅)
        xs, ys = zip(*ref_points)
        n = len(xs)
        x_mean, y_mean = stats.mean(xs), stats.mean(ys)
        a = sum((x - x_mean)*(y - y_mean) for x, y in ref_points) / sum((x - x_mean)**2 for x in xs)
        b = y_mean - a * x_mean

        # 관계: kg ≈ a*raw + b
        # scale = 1/a, offset = -b/a
        scale = 1.0 / a
        offset = -b / a

        cfg = {"data_pin": DATA_PIN, "clk_pin": CLK_PIN,
               "offset": offset, "ref_unit": scale}
        with open(CALIB_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
        print("Calibration saved:", cfg)

if __name__ == "__main__":
    main()
