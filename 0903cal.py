import time, json, statistics as stats
from RpiHX711 import HX711

CALIB_PATH = "hx711_calibration.json"

def mean_raw(hx, n=20, delay_s=0.05):
    vals = []
    for _ in range(n):
        vals.append(hx.get_raw_data_mean(readings=1))
        time.sleep(delay_s)
    return stats.mean(vals)

def calibrate(hx, known_mass_kg, n=30):
    print("Step 1: Empty the scale.")
    input("Press Enter when ready...")
    offset = mean_raw(hx, n=n)

    print(f"Offset(raw) = {offset}")

    print(f"Step 2: Place {known_mass_kg} kg weight.")
    input("Press Enter when ready...")
    raw_with_weight = mean_raw(hx, n=n)

    delta = raw_with_weight - offset
    if abs(delta) < 1:
        raise RuntimeError("Calibration failed: no difference detected.")

    scale = delta / known_mass_kg  # counts per kg
    cfg = {"offset": offset, "scale": scale}
    with open(CALIB_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

    print("Calibration saved:", cfg)

if __name__ == "__main__":
    hx = HX711(dout=5, pd_sck=6)  # 핀 번호 맞게 수정
    hx.reset()
    calibrate(hx, known_mass_kg=1.000)
