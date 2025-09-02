#!/usr/bin/env python3
# hx711_calibrate_endail.py
# endail/hx711 Python 바인딩용: ref_unit/offset 보정값을 JSON에 저장

import json, time
from HX711 import *   # SimpleHX711, Mass

DATA_PIN = 5
CLK_PIN  = 6
CALIB_PATH = "hx711_calibration.json"

def read_weight_num(hx, samples=35, repeats=5, delay=0.05):
    vals = []
    for _ in range(repeats):
        w = hx.weight(samples)   # 숫자(float)여야 함
        try:
            w = float(w)         # 혹시 객체면 float 캐스팅
        except Exception:
            w = getattr(w, "value", w)  # 마지막 안전장치
        vals.append(w)
        time.sleep(delay)
    return sum(vals) / len(vals)

def main():
    # ref_unit/offset은 임시값으로 시작해도 됨 (예: 1, 0)
    with SimpleHX711(DATA_PIN, CLK_PIN, 1, 0) as hx:
        hx.setUnit(Mass.Unit.KG)

        print("Step 1: Remove all weight.")
        input("Press Enter when ready...")
        hx.zero()                        # 내부 offset 갱신
        offset = hx.getOffset()          # 보정에 쓸 offset 기록
        print("Offset =", offset)

        print("Step 2: Place known weight (e.g. 1.000 kg).")
        input("Press Enter when ready...")

        measured = read_weight_num(hx, samples=35, repeats=6)
        print(f"Measured (with current ref_unit) = {measured:.6f} kg")

        # 현재 ref_unit을 가져와서, 비례식으로 보정
        cur_ref = hx.getReferenceUnit()
        # weight ≈ (raw - offset) / ref_unit 이라 가정 → 출력은 ref_unit에 반비례
        # 원하는 출력 known_mass에 맞추려면: new_ref = cur_ref * (measured / known)
        known_mass = float(input("Enter known mass in kg (e.g. 1.0): ").strip() or "1.0")
        if known_mass <= 0:
            raise ValueError("Known mass must be positive.")

        new_ref = cur_ref * (measured / known_mass)
        hx.setReferenceUnit(new_ref)

        # 확인 읽기
        verified = read_weight_num(hx, samples=35, repeats=6)
        print(f"After setReferenceUnit({new_ref:.6f}), verify = {verified:.6f} kg")

        # 저장
        cfg = {"data_pin": DATA_PIN, "clk_pin": CLK_PIN,
               "offset": offset, "ref_unit": new_ref}
        with open(CALIB_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
        print("Saved calibration:", cfg)

if __name__ == "__main__":
    main()
