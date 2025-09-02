#!/usr/bin/env python3
import time, json
from HX711 import *

CALIB_PATH = "hx711_calibration.json"

def main():
    with open(CALIB_PATH, "r") as f:
        cfg = json.load(f)

    with SimpleHX711(cfg["data_pin"], cfg["clk_pin"], 1, 0) as hx:
        hx.setUnit(Mass.Unit.KG)
        a, b = cfg["a"], cfg["b"]

        while True:
            measured = float(hx.weight(35))
            corrected = a * measured + b
            print(f"{corrected:.3f} kg")
            time.sleep(1)

if __name__ == "__main__":
    main()
