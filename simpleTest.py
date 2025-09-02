#!/usr/bin/env python3
# realtime_hx711_min.py

import time
import RPi.GPIO as GPIO
from hx711 import HX711

# --- BCM pin numbers: edit to match your wiring ---
DOUT_PIN = 5      # HX711 DOUT  -> Raspberry Pi BCM 5  (example)
SCK_PIN  = 6      # HX711 PD_SCK-> Raspberry Pi BCM 6  (example)

def main():
    hx = HX711(dout_pin=DOUT_PIN, pd_sck_pin=SCK_PIN, channel='A', gain=128)
    hx.reset()  # optional

    # quick "tare": capture baseline with nothing on the load cell
    print("Capturing baseline (tare)...")
    baseline = sum(hx.get_raw_data(times=10)) / 10.0
    print(f"Baseline (raw avg): {baseline:.0f}")

    print("Reading...  Ctrl+C to stop.")
    while True:
        vals = hx.get_raw_data(times=5)          # read a few samples
        avg  = sum(vals) / len(vals)
        raw_delta = avg - baseline               # zeroed raw value

        # If you have a scale factor, apply it here (kg_per_count)
        # weight_kg = raw_delta * KG_PER_COUNT

        print(f"raw_avg={avg:.0f}  delta={raw_delta:.0f}")
        time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
