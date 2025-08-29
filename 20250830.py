#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HX711 + OpenCV Sequencer
- Keeps adaptive EMA + Zero-Lock + Z/C controls
- Sequences:
  01(wait) -> 02(measure twice, avg) -> 03/03-1(result pause/resume) -> 01
- NEW: In seq02, if weight <= 20000g at any time, jump to seq01 immediately.
"""

import cv2, time, threading, signal, sys
from hx711 import HX711
import RPi.GPIO as GPIO

# -------------------------------
# HX711 / GPIO
# -------------------------------
GPIO.setmode(GPIO.BCM)
hx = HX711(dout_pin=5, pd_sck_pin=6)  # DOUT=5, SCK=6

# -------------------------------
# USER SETTINGS
# -------------------------------
WINDOW_NAME   = "Player"
HEADLESS      = False
SPEED_SCALE   = 1.0

# Calibration slope (press 'C' to recalibrate with known mass)
A = 0.03716
CAL_MASS_G = 69200.0

# Sampling / smoothing
ZERO_SAMPLES  = 60
READ_SAMPLES  = 6
EMA_ALPHA_SLOW = 0.18
EMA_ALPHA_FAST = 0.60
DELTA_RAW_FAST = 250
PRINT_EVERY    = 0.5
LOOP_SLEEP     = 0.02

# Zero-Lock
ZERO_LOCK_THRESHOLD = 5.0  # g
ZERO_LOCK_MIN_TIME  = 0.3  # s

# Sequence thresholds
TRIGGER_MIN  = 20000.0      # >=
TRIGGER_MAX  = 120000.0     # <
BRANCH_SPLIT = 65000.0      # seq02 avg split
PAUSE_UNDER  = 20000.0      # < for 2~3 seconds (here used in seq03 pause condition)
PAUSE_HOLD_S = 3.0

# -------------------------------
# Shared state / events
# -------------------------------
stop_event    = threading.Event()
zero_request  = threading.Event()
cal_request   = threading.Event()
state_lock    = threading.Lock()

zero_raw = None          # updated by (re)zero
cur_weight_g = 0.0       # latest filtered weight (for UI/logic)
curA = A                 # safe copy for printing

def _install_sig_handlers():
    def _h(sig, frame): stop_event.set()
    signal.signal(signal.SIGINT, _h); signal.signal(signal.SIGTERM, _h)

# ------------- HX711 helpers -------------
def _measure_zero_raw():
    return hx.get_raw_data_mean(ZERO_SAMPLES)

def _auto_zero(tag="start") -> bool:
    global zero_raw
    raw0 = _measure_zero_raw()
    if raw0 is None:
        print(f
