#!/usr/bin/env python3
# hx711_simple.py
# Minimal HX711 bit-bang reader using RPi.GPIO

import time
import RPi.GPIO as GPIO


class HX711Simple:
    """Minimal HX711 reader (bit-banged) for Raspberry Pi (BCM numbering)."""

    def __init__(
        self,
        dout_pin: int,
        sck_pin: int,
        channel_select: str = "A",      # "A" or "B"
        channel_A_gain: int = 128,      # 128 or 64 (for channel A). Ignored if channel=B
        pulse_delay_s: float = 0.000002 # ~2us pulse width
    ):
        # Save config
        self.dout = dout_pin
        self.sck = sck_pin
        self.channel_select = channel_select.upper()
        self.channel_A_gain = int(channel_A_gain)
        self.pulse_delay_s = float(pulse_delay_s)

        # Init GPIO mode once if needed
        if GPIO.getmode() is None:
            GPIO.setmode(GPIO.BCM)

        # Setup pins
        GPIO.setup(self.sck, GPIO.OUT)
        GPIO.setup(self.dout, GPIO.IN)

        # Ensure clock low to start
        GPIO.output(self.sck, False)

    # ------------- Low-level helpers -------------
    def _pulse(self):
        """Generate one clock pulse to HX711."""
        GPIO.output(self.sck, True)
        if self.pulse_delay_s > 0:
            time.sleep(self.pulse_delay_s)
        GPIO.output(self.sck, False)
        if self.pulse_delay_s > 0:
            time.sleep(self.pulse_delay_s)

    def is_ready(self) -> bool:
        """HX711 is ready when DOUT goes low."""
        return GPIO.input(self.dout) == 0

    def wait_ready(self, timeout_s: float = 0.5) -> bool:
        """Wait until DOUT==0 or timeout."""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.is_ready():
                return True
            time.sleep(0.0005)  # 0.5ms poll
        return False

    def _post_read_set_channel(self):
        """After 24-bit read, send extra pulses to select next channel/gain."""
        # Default is Channel B (2 pulses)
        num_pulses = 2
        if self.channel_select == "A" and self.channel_A_gain == 128:
            num_pulses = 1
        elif self.channel_select == "A" and self.channel_A_gain == 64:
            num_pulses = 3
        for _ in range(num_pulses):
            self._pulse()

    # ------------- Public API -------------
    def read_raw(self, timeout_s: float = 0.5) -> int:
        """
        Read 24-bit raw value (two's complement format, not yet converted).
        Returns the 24-bit integer (0..0xFFFFFF), or raises RuntimeError on failure.
        """
        if not self.wait_ready(timeout_s=timeout_s):
            raise RuntimeError("HX711 not ready (DOUT stayed high).")

        raw = 0
        for _ in range(24):
            self._pulse()
            # Shift left and OR with current DOUT bit
            raw = (raw << 1) | GPIO.input(self.dout)

        # Set channel/gain for next conversion
        self._post_read_set_channel()

        return raw & 0xFFFFFF

    def read_signed(self, timeout_s: float = 0.5) -> int:
        """
        Read value and convert from HX711 24-bit two's complement to signed int.
        Also flags known invalid sentinel values.
        """
        raw = self.read_raw(timeout_s=timeout_s)

        # Check invalid sentinel values
        if raw in (0x000000, 0x800000, 0x7FFFFF, 0xFFFFFF):
            # These often indicate wiring/noise/out-of-range
            # Still return converted value, but raise a warning via print.
            print("[Warn] Invalid/sentinel raw value:", hex(raw))

        # Convert 24-bit two's complement to Python int
        if raw & 0x800000:  # sign bit set => negative
            signed = -((raw ^ 0xFFFFFF) + 1)
        else:
            signed = raw
        return signed

    def power_down(self):
        """Put HX711 into power-down by holding SCK high for >60us."""
        GPIO.output(self.sck, False)
        if self.pulse_delay_s > 0:
            time.sleep(self.pulse_delay_s)
        GPIO.output(self.sck, True)
        time.sleep(0.0001)  # >60us
        # Keep SCK high while powered down

    def power_up(self):
        """Wake HX711 by pulling SCK low."""
        GPIO.output(self.sck, False)
        if self.pulse_delay_s > 0:
            time.sleep(self.pulse_delay_s)

    def cleanup(self):
        """Release GPIO resources (affects global RPi.GPIO)."""
        GPIO.cleanup()


# -----------------------------------------------
# Example (remove or comment out in production)
# -----------------------------------------------
if __name__ == "__main__":
    # Example usage: DOUT=2, SCK=1, Channel A @ gain 128
    hx = HX711Simple(dout_pin=2, sck_pin=1, channel_select="A", channel_A_gain=128)

    try:
        val_raw = hx.read_raw()
        print(f"Raw read (24-bit): {val_raw:#08x}")

        val_signed = hx.read_signed()
        print(f"Raw read (signed integer): {val_signed}")

    except Exception as e:
        print("[Error]", e)
    finally:
        hx.cleanup()
