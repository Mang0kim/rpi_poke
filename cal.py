from HX711 import *
import time, statistics as stats

def mean_weight(hx, n=20):
    vals = []
    for _ in range(n):
        vals.append(hx.weight())
        time.sleep(0.05)
    return stats.mean(vals)

with SimpleHX711(2, 3, 1, 0) as hx:  # ref_unit=1, offset=0 임시
    hx.setUnit(Mass.Unit.KG)

    print("Step 1: Remove all weight.")
    input("Press Enter when ready...")
    hx.zero()
    offset = hx.getOffset()
    print("Offset =", offset)

    print("Step 2: Place known weight (e.g. 1.000 kg).")
    input("Press Enter when ready...")
    measured = mean_weight(hx)
    known_mass = 1.000  # kg
    ref_unit = (measured - offset) / known_mass
    print("Ref unit =", ref_unit)

    print("Calibration done. Use:")
    print(f"SimpleHX711(2, 3, {ref_unit:.0f}, {offset:.0f})")
