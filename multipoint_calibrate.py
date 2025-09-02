#!/usr/bin/env python3
import json, time, os, statistics as stats
from HX711 import *

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(BASE_DIR, "hx711_calibration.json")
MODEL_PATH = os.path.join(BASE_DIR, "hx711_model.json")

# 샘플링 파라미터
SAMPLES_PER_READ = 10     # hx.weight(10) 한 번 호출
REPEATS_PER_POINT = 8     # 각 기준점에서 반복 횟수
SLEEP_BETWEEN = 0.10      # 반복 간 대기

def robust_point(hx, repeats=REPEATS_PER_POINT):
    vals = []
    for _ in range(repeats):
        try:
            v = float(hx.weight(SAMPLES_PER_READ))
            vals.append(v)
        except Exception:
            pass
        time.sleep(SLEEP_BETWEEN)
    if not vals:
        return float("nan")
    # 트림드 평균(상하 1개 제거) + 평균, 표본 적으면 그냥 평균
    vals.sort()
    if len(vals) >= 5:
        trimmed = vals[1:-1]
        return sum(trimmed) / len(trimmed)
    return sum(vals) / len(vals)

def build_piecewise(points):
    """
    points: list of dicts [{"measured": m, "actual": a}, ...], measured 오름차순
    반환: segments = [{"m0":..., "m1":..., "slope":..., "intercept":...}, ...]
    실제값 = slope * measured + intercept
    """
    pts = sorted(points, key=lambda x: x["measured"])
    segs = []
    for i in range(len(pts) - 1):
        m0, a0 = pts[i]["measured"], pts[i]["actual"]
        m1, a1 = pts[i+1]["measured"], pts[i+1]["actual"]
        if abs(m1 - m0) < 1e-9:
            slope, itc = 1.0, 0.0
        else:
            slope = (a1 - a0) / (m1 - m0)
            itc   = a0 - slope * m0
        segs.append({"m0": m0, "m1": m1, "slope": slope, "intercept": itc})
    return pts, segs

def main():
    if not os.path.isfile(CALIB_PATH):
        raise FileNotFoundError(f"{CALIB_PATH} not found.")

    with open(CALIB_PATH, "r") as f:
        cfg = json.load(f)

    print("=== HX711 Multipoint Calibration ===")
    print("무게를 쉼표로 입력 (예: 1, 1.75, 5, 10).")
    print("빈 입력이면 종료합니다.")
    raw = input("기준 무게(kg)들: ").strip()

    if not raw:
        print("입력 없음. 종료.")
        return

    try:
        targets = [float(x) for x in raw.replace("kg","").split(",") if x.strip() != ""]
    except Exception:
        print("파싱 실패. 예: 1, 1.75, 5")
        return

    points = []
    with SimpleHX711(cfg["data_pin"], cfg["clk_pin"], 1, 0) as hx:
        hx.setUnit(Mass.Unit.KG)

        # 0kg 포인트(빈 저울)도 권장
        if 0.0 not in targets:
            print("\n[권장] 빈 저울 0kg 포인트도 추가합니다.")
            targets = [0.0] + targets

        for kg in targets:
            input(f"\n>>> {kg} kg 올리고 Enter를 누르세요...")
            measured = robust_point(hx)
            print(f"measured={measured:.6f}  actual={kg:.6f}")
            points.append({"measured": measured, "actual": kg})

    # 최소 2점 필요
    if len(points) < 2:
        print("기준점이 2개 미만입니다. 취소.")
        return

    pts_sorted, segments = build_piecewise(points)
    model = {
        "model": "piecewise",
        "points": pts_sorted,
        "segments": segments,
        "meta": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "repeats_per_point": REPEATS_PER_POINT,
            "samples_per_read": SAMPLES_PER_READ
        }
    }

    with open(MODEL_PATH, "w") as f:
        json.dump(model, f, indent=2)

    print(f"\n저장 완료: {MODEL_PATH}")
    print("segments:")
    for s in segments:
        print(f"  [{s['m0']:.3f} .. {s['m1']:.3f}]  actual = {s['slope']:.6f} * measured + {s['intercept']:.6f}")

if __name__ == "__main__":
    main()
