from HX711 import *

# create a SimpleHX711 object using GPIO pin 2 as the data pin,
# GPIO pin 3 as the clock pin, -370 as the reference unit, and
# -367471 as the offset
with SimpleHX711(2, 3, -370, -367471) as hx:

  # set the scale to output weights in ounces
  hx.setUnit(Mass.Unit.OZ)

  # zero the scale
  hx.zero()

  # constantly output weights using the median of 35 samples
  while True:
    print(hx.weight(35)) #eg. 1.08 oz
