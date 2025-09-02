#!/usr/bin/env python3
import time, json, statistics as stats
from HX711 import *

CALIB_PATH = "hx711_calibration.json"
DATA_PIN = 5
CLK_PIN  = 6

def mean_weight(hx, samples=35, repeats=10, delay=0.05):
    vals = []
    for _ in range(repeats):
        v = float(hx.weight(samples))   # 무조건 float 캐스팅
        vals.append(v)
        time.sleep(delay)
    return stats.mean(vals)

def main():
    with SimpleHX711(DATA_PIN, CLK_PIN, 1, 0) as hx:
        hx.setUnit(Mass.Unit.KG)

        ref_points = []
        for known_mass in [0.0, 1.75, 70.0]:
            print(f"Place {known_mass} kg weight.")
            input("Press Enter when ready...")
            measured = mean_weight(hx)
            print(f"Measured at {known_mass} kg = {measured:.6f}")
            ref_points.append((measured, known_mass))

        # measured → actual 관계에 대해 선형회귀: actual ≈ a*measured + b
        xs, ys = zip(*ref_points)
        n = len(xs)
        x_mean, y_mean = stats.mean(xs), stats.mean(ys)
        a = sum((x - x_mean)*(y - y_mean) for x, y in ref_points) / sum((x - x_mean)**2 for x in xs)
        b = y_mean - a * x_mean

        cfg = {"data_pin": DATA_PIN, "clk_pin": CLK_PIN, "a": a, "b": b}
        with open(CALIB_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
        print("Calibration saved:", cfg)

if __name__ == "__main__":
    main()
