#!/usr/bin/env python3
import time, os, sys
import RPi.GPIO as GPIO
from hx711 import HX711

DOUT_PIN = 5   # HX711 DT
SCK_PIN  = 6   # HX711 SCK

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

hx = HX711(dout_pin=DOUT_PIN, pd_sck_pin=SCK_PIN)

def restart_program():
    """재시작 함수"""
    GPIO.cleanup()
    python = sys.executable
    os.execl(python, python, *sys.argv)

try:
    while True:
        try:
            val = hx.get_raw_data_mean(readings=1)
            if val is not False:
                print(val)
            else:
                print("invalid")
        except Exception as e:
            print("Error:", e, " → 프로그램 재시작")
            restart_program()
        time.sleep(0.1)

finally:
    GPIO.cleanup()
