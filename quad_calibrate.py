#!/usr/bin/env python3
import os, time, json, pathlib
from HX711 import *
from statistics import median

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(BASE_DIR, "hx711_calibration.json")  # data_pin/clk_pin만 사용
QUAD_PATH  = os.path.join(BASE_DIR, "hx711_quad.json")

def load_calib():
    with open(CALIB_PATH, "r") as f:
        return json.load(f)

def read_measured(hx, seconds=1.2, gap_s=0.02, group=4):
    """robust_read_kg와 동일 개념: 묶음평균의 중앙값"""
    buf, grp = [], []
    t_end = time.time() + seconds
    while time.time() < t_end:
        try:
            w = float(hx.weight(1))
            grp.append(w)
            if len(grp) >= group:
                buf.append(sum(grp)/len(grp))
                grp = []
        except Exception:
            pass
        time.sleep(gap_s)
    if grp: buf.append(sum(grp)/len(grp))
    buf.sort()
    return buf[len(buf)//2] if buf else float('nan')

def solve_quadratic(x1,y1, x2,y2, x3,y3):
    """
    세 점(xi, yi)을 지나는 2차식 y = c2 x^2 + c1 x + c0 의 계수 계산.
    Lagrange 보간 기반(안정적이고 외부 라이브러리 불필요).
    """
    def L0(x):  # (x - x2)(x - x3) / ((x1 - x2)(x1 - x3))
        return lambda z: ((z - x2)*(z - x3))/((x1 - x2)*(x1 - x3))
    def L1(x):  # (x - x1)(x - x3) / ((x2 - x1)(x2 - x3))
        return lambda z: ((z - x1)*(z - x3))/((x2 - x1)*(x2 - x3))
    def L2(x):  # (x - x1)(x - x2) / ((x3 - x1)(x3 - x2))
        return lambda z: ((z - x1)*(z - x2))/((x3 - x1)*(x3 - x2))

    # y(z) = y1*L0(z) + y2*L1(z) + y3*L2(z)
    # 이를 전개해 z^2, z, 상수항의 계수(c2, c1, c0)를 구함
    import math
    # 각 Lk(z) = A_k*z^2 + B_k*z + C_k 형태로 전개
    def poly_of_L(L):
        # L(z) = K*(z - a)*(z - b) = K*(z^2 - (a+b)z + ab)
        # 여기서 K, a, b를 읽어내는 대신 수치적으로 계수 구함
        # z^2, z, 1 의 계수를 z=0,1,2 대입해서 해도 되지만 정확도를 위해 직접 전개: 
        # 아래 방식은 파라메터를 알기 어렵기 때문에 샘플링 전개로 충분 (정수 3점이면 완전결정)
        # 샘플 포인트 3개로 y = c2 z^2 + c1 z + c0 를 푼다.
        Z = [0.0, 1.0, 2.0]
        Y = [L(zz) for zz in Z]
        # 3x3 선형계 풀기
        # [0^2 0 1][c2 c1 c0]^T = y0
        # [1   1 1][...] = y1
        # [4   2 1][...] = y2
        y0, y1, y2 = Y
        # 행렬 해(수작업):
        # from solving quickly:
        # c0 = y0
        c0 = y0
        # y1 = c2*1 + c1*1 + c0
        # y2 = c2*4 + c1*2 + c0
        # => (y1-c0) = c2 + c1
        #    (y2-c0) = 4c2 + 2c1
        A = y1 - c0
        B = y2 - c0
        # 2*(A) = 2c2 + 2c1
        # B - 2A = (4c2 + 2c1) - (2c2 + 2c1) = 2c2
        c2 = 0.5*(B - 2*A)
        c1 = A - c2
        return c2, c1, c0

    L0p = L0(None); L1p = L1(None); L2p = L2(None)
    c20, c10, c00 = poly_of_L(L0p)
    c21, c11, c01 = poly_of_L(L1p)
    c22, c12, c02 = poly_of_L(L2p)

    # 최종 계수 조합
    C2 = y1*c20 + y2*c21 + y3*c22
    C1 = y1*c10 + y2*c11 + y3*c12
    C0 = y1*c00 + y2*c01 + y3*c02
    return C2, C1, C0

def main():
    cfg = load_calib()
    data_pin, clk_pin = cfg["data_pin"], cfg["clk_pin"]

    print("[*] Quadratic calibration")
    print("0kg, 70kg, 105kg state holding and Enter.")
    with SimpleHX711(data_pin, clk_pin, 1, 0) as hx:
        hx.setUnit(Mass.Unit.KG)  # 현재 코드와 동일 기준으로 'measured'를 얻는다

        actuals = [0.0, 70.0, 105.0]
        measureds = []

        for tgt in actuals:
            input(f" -> {tgt:.0f} kg state check and Enter.")
            m = read_measured(hx, seconds=1.2)
            print(f"    measured = {m:.6f}")
            measureds.append(m)

    (x1,y1), (x2,y2), (x3,y3) = (measureds[0], actuals[0]), (measureds[1], actuals[1]), (measureds[2], actuals[2])
    c2, c1, c0 = solve_quadratic(x1,y1, x2,y2, x3,y3)

    model = {
        "model": "quadratic",
        "points": [
            {"measured": x1, "actual": y1},
            {"measured": x2, "actual": y2},
            {"measured": x3, "actual": y3}
        ],
        "coeffs": {"c2": c2, "c1": c1, "c0": c0}
    }
    pathlib.Path(QUAD_PATH).write_text(json.dumps(model, indent=2))
    print(f"[OK] Saved quadratic model -> {QUAD_PATH}")
    print(f"     kg = ({c2:.10g})*m^2 + ({c1:.10g})*m + ({c0:.10g})")

if __name__ == "__main__":
    main()
