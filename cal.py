#!/usr/bin/env python3
# hx711_calibrate.py
# Raspberry Pi + HX711 캘리브레이션 전용 스크립트
# 1) 영점(오프셋) 측정 → 2) 기준추 무게 입력 → 3) 스케일 저장(JSON)

import time
import json
import math
import argparse
from statistics import median
import RPi.GPIO as GPIO

# ---------- 사용자 설정 (배선에 맞게 수정) ----------
DOUT_PIN = 2        # BCM 번호 (예: 2)
SCK_PIN  = 3        # BCM 번호 (예: 3)
CHANNEL_SELECT = "A"     # "A" 또는 "B"
CHANNEL_A_GAIN = 128     # 128 또는 64 (A채널일 때만)
PULSE_DELAY_S  = 2e-6    # SCK 펄스폭 (~2us)
CALIB_PATH = "hx711_calibration.json"

# 샘플링/안정화 파라미터
N_SAMPLES_OFFSET = 25    # 영점 샘플 개수
N_SAMPLES_WEIGHT = 25    # 기준추 샘플 개수
STABLE_WINDOW = 7        # 안정성 판단에 사용할 최근 샘플 개수
STABLE_TOL = 6           # 안정성 허용 편차(원시 카운트). 배선/노이즈에 따라 조정

# ---------- 최소 HX711 리더 ----------
class HX711Simple:
    """Minimal HX711 reader (bit-banged) for Raspberry Pi (BCM)."""

    def __init__(self, dout_pin, sck_pin,
                 channel_select="A", channel_A_gain=128,
                 pulse_delay_s=2e-6):
        self.dout = dout_pin
        self.sck = sck_pin
        self.channel_select = channel_select.upper()
        self.channel_A_gain = int(channel_A_gain)
        self.pulse_delay_s = float(pulse_delay_s)

        if GPIO.getmode() is None:
            GPIO.setmode(GPIO.BCM)

        GPIO.setup(self.sck, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.dout, GPIO.IN)

        # 프리시리즈: 원하는 채널/게인으로 맞추기 위해 더미 리드
        self._channel_pulses()

    def _channel_pulses(self):
        """채널/게인 설정을 위한 추가 SCK 펄스 수 계산."""
        if self.channel_select == "A":
            return 1 if self.channel_A_gain == 128 else 3  # 128→1, 64→3
        else:
            return 2  # 채널 B

    def is_ready(self, tries=20):
        for _ in range(tries):
            if GPIO.input(self.dout) == 0:
                return True
            time.sleep(0.001)
        return False

    def _pulse(self):
        GPIO.output(self.sck, True)
        time.sleep(self.pulse_delay_s)
        GPIO.output(self.sck, False)
        time.sleep(self.pulse_delay_s)

    def read_raw(self, timeout_s=0.4):
        """24비트 원시값 읽기. 실패 시 None."""
        t0 = time.time()
        while GPIO.input(self.dout) != 0:
            if (time.time() - t0) > timeout_s:
                return None  # 데이터 준비 안 됨
            time.sleep(0.0005)

        value = 0
        for _ in range(24):
            self._pulse()
            value = (value << 1) | GPIO.input(self.dout)

        # 채널/게인 세팅용 추가 펄스
        for _ in range(self._channel_pulses()):
            self._pulse()

        # 24비트 2의 보수 처리
        if value & 0x800000:
            value -= 1 << 24
        return value

    def read_median(self, n=7, allow_none=False):
        vals = []
        for _ in range(n):
            v = self.read_raw()
            if v is None:
                if allow_none:
                    continue
                else:
                    # 짧은 휴식 후 재시도
                    time.sleep(0.003)
                    v = self.read_raw()
            if v is not None:
                vals.append(v)
            time.sleep(0.002)
        if not vals:
            return None
        return int(median(vals))

    def cleanup(self):
        GPIO.cleanup(self.sck)
        GPIO.cleanup(self.dout)

# ---------- 유틸 ----------
def wait_for_stable(hx: HX711Simple, label="no-load", window=STABLE_WINDOW, tol=STABLE_TOL, timeout_s=10):
    """최근 window개의 원시값 편차가 tol 이하가 될 때까지 대기."""
    ring = []
    t0 = time.time()
    while True:
        v = hx.read_raw()
        if v is not None:
            ring.append(v)
            if len(ring) > window:
                ring.pop(0)
            if len(ring) == window:
                span = max(ring) - min(ring)
                print(f"[{label}] last{window} span={span}  mid={int(sum(ring)/len(ring))}")
                if span <= tol:
                    return int(sum(ring)/len(ring))
        if (time.time() - t0) > timeout_s:
            print(f"[Warn] Stability timeout on {label}, using current median.")
            mv = hx.read_median(n=window*2, allow_none=True)
            return mv if mv is not None else 0

def average_samples(hx: HX711Simple, n):
    vals = []
    for _ in range(n):
        v = hx.read_raw()
        if v is not None:
            vals.append(v)
        time.sleep(0.002)
    if not vals:
        raise RuntimeError("No valid samples. Check wiring/supply/noise.")
    return sum(vals) / len(vals)

def save_calibration(path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved calibration → {path}")

# ---------- 메인 캘리브레이션 절차 ----------
def calibrate(known_weight_g: float):
    hx = HX711Simple(DOUT_PIN, SCK_PIN,
                     channel_select=CHANNEL_SELECT,
                     channel_A_gain=CHANNEL_A_GAIN,
                     pulse_delay_s=PULSE_DELAY_S)
    try:
        print("=== HX711 Calibration ===")
        print("1) 아무것도 올리지 마세요. 영점 안정화 중...")
        offset_stable = wait_for_stable(hx, label="offset")
        offset_avg = average_samples(hx, N_SAMPLES_OFFSET)
        offset = int(round((offset_stable + offset_avg) / 2.0))
        print(f"[Step1] Offset (no-load) = {offset}")

        input("\n2) 기준추를 올려주세요. 올렸으면 Enter ▶ ")

        print("   기준추 안정화 중...")
        weight_stable = wait_for_stable(hx, label="weight")
        weight_avg = average_samples(hx, N_SAMPLES_WEIGHT)
        raw_with_weight = int(round((weight_stable + weight_avg) / 2.0))
        print(f"[Step2] Raw(with weight) = {raw_with_weight}")

        delta = raw_with_weight - offset
        if delta == 0:
            raise RuntimeError("Delta is zero. Check that a weight is actually placed.")

        # 스케일: (raw - offset) / grams  →  1 gram 당 카운트
        scale_counts_per_g = delta / known_weight_g
        # g/카운트도 같이 저장(필요 시 역수)
        scale_g_per_count = 1.0 / scale_counts_per_g

        calib = {
            "dout_pin": DOUT_PIN,
            "sck_pin": SCK_PIN,
            "channel_select": CHANNEL_SELECT,
            "channel_A_gain": CHANNEL_A_GAIN,
            "offset": int(offset),
            "scale_counts_per_g": float(scale_counts_per_g),
            "scale_g_per_count": float(scale_g_per_count),
            "known_weight_g": float(known_weight_g),
            "raw_with_weight": int(raw_with_weight),
            "raw_offset_source": {
                "stable_est": int(offset_stable),
                "avg_est": int(offset_avg)
            },
            "notes": "weight(g) ≈ (raw - offset) * scale_g_per_count"
        }
        save_calibration(CALIB_PATH, calib)

        # 빠른 검증
        test_raw = hx.read_median(n=9, allow_none=True)
        if test_raw is not None:
            est_g = (test_raw - offset) * scale_g_per_count
            print(f"[Check] instantaneous raw={test_raw}, estimated ≈ {est_g:.2f} g")

        print("\n완료! 이후 측정 식:")
        print("  weight_g ≈ (raw - offset) * scale_g_per_count")
        print(f"  offset={offset}, scale_g_per_count={scale_g_per_count:.8f}")

    finally:
        hx.cleanup()
        GPIO.cleanup()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="HX711 Calibration (offset & scale)")
    ap.add_argument("--known", type=float, required=True,
                    help="기준추 무게 (gram). 예: --known 1000")
    args = ap.parse_args()
    calibrate(args.known)

if __name__ == "__main__":
    main()
