#!/usr/bin/env python3
import time
import RPi.GPIO as GPIO
from hx711 import HX711

# --- 핀 번호(BCM) ---
DOUT_PIN = 5   # HX711 DT
SCK_PIN  = 6   # HX711 SCK

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

hx = HX711(dout_pin=DOUT_PIN, pd_sck_pin=SCK_PIN)

try:
    while True:
        val = hx.get_raw_data_mean(readings=10)  # 평균 10회
        # 라이브러리는 실패 시 False를 반환할 수 있음
        print(val if val is not False else "invalid")
        time.sleep(0.1)
finally:
    GPIO.cleanup()
