from HX711 import SimpleHX711, Rate
import time

with SimpleHX711(5, 6, -370, -200000, Rate.HZ_10) as hx:
    while True:
        print(hx.weight())
        time.sleep(0.1)
