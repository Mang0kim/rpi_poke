#!/usr/bin/env python3
# enhanced_weight_system.py
import os, cv2, time, json, statistics, threading, logging
import numpy as np
from collections import deque
from typing import Deque, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from hx711 import HX711
import RPi.GPIO as GPIO

# ---------------- Configuration ----------------
@dataclass
class SystemConfig:
    """Centralized configuration"""
    # Paths
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    video_files: dict = None
    calibration_file: str = "calibration.json"
    
    # Weight thresholds
    trigger_threshold_kg: float = 0.8
    stability_window_sec: float = 3.0
    stability_band_kg: float = 1.5
    baseline_epsilon_kg: float = 0.1
    
    # Video settings
    display_fps: int = 30
    seq3_pause_time_sec: float = 2.0
    
    # Sensor settings
    ema_alpha: float = 0.25
    reads_per_sample: int = 10
    read_delay: float = 0.004
    read_hz: float = 15.0
    warmup_sec: float = 1.0
    snapshot_sec: float = 0.8
    
    # Hardware pins
    dt_pin: int = 5
    sck_pin: int = 6
    gain: int = 128
    
    def __post_init__(self):
        if self.video_files is None:
            self.video_files = {
                'idle': os.path.join(self.base_dir, "01.mp4"),
                'measuring': os.path.join(self.base_dir, "02.mp4"),
                'confirmed_normal': os.path.join(self.base_dir, "03.mp4"),
                'confirmed_heavy': os.path.join(self.base_dir, "03-1.mp4")
            }

class SequenceState(Enum):
    IDLE = "idle"
    MEASURING = "measuring" 
    CONFIRMED = "confirmed"
    ABORT = "abort"

# ---------------- Enhanced Logging ----------------
def setup_logging():
    """Setup structured logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('weight_system.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ---------------- Calibration with Validation ----------------
class WeightCalibration:
    def __init__(self, config_path: str):
        self.model, self.coeffs, self.polarity = self._load_and_validate(config_path)
    
    def _load_and_validate(self, path: str):
        """Load calibration with validation"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            model = data.get("model", "linear")
            coeffs = data["coeffs"]
            polarity = float(data.get("polarity", 1.0))
            
            if model not in ("linear", "quadratic"):
                raise ValueError(f"Invalid model: {model}")
            
            # Validate coefficient structure
            required_keys = {"a1", "b1"} if model == "linear" else {"a2", "b2", "c2"}
            if not required_keys.issubset(coeffs.keys()):
                raise ValueError(f"Missing coefficients for {model} model")
                
            return model, coeffs, polarity
            
        except Exception as e:
            raise RuntimeError(f"Failed to load calibration: {e}")
    
    def predict_weight(self, raw_value: float) -> float:
        """Convert raw sensor value to weight"""
        x = self.polarity * raw_value
        if self.model == "linear":
            return self.coeffs["a1"] * x + self.coeffs["b1"]
        else:
            return self.coeffs["a2"] * x * x + self.coeffs["b2"] * x + self.coeffs["c2"]

# ---------------- Enhanced Weight Reader ----------------
class WeightReader:
    def __init__(self, config: SystemConfig, calibration: WeightCalibration):
        self.config = config
        self.calibration = calibration
        self.hx = HX711(
            dout_pin=config.dt_pin,
            pd_sck_pin=config.sck_pin,
            channel='A',
            gain=config.gain
        )
        self.ema = EMA(config.ema_alpha)
        self.current_weight = 0.0
        self.history: Deque[Tuple[float, float]] = deque(maxlen=600)
        self.baseline_ready = threading.Event()
        self.stop_reading = threading.Event()
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    def _robust_read(self) -> float:
        """Read sensor with statistical filtering"""
        vals = []
        for _ in range(self.config.reads_per_sample):
            try:
                raw_data = self.hx.get_raw_data()
                if isinstance(raw_data, (int, float)):
                    vals.append(float(raw_data))
                elif isinstance(raw_data, (list, tuple)):
                    vals.extend([float(v) for v in raw_data if isinstance(v, (int, float))])
            except Exception as e:
                self.logger.warning(f"Sensor read error: {e}")
            time.sleep(self.config.read_delay)
            
        if not vals:
            raise RuntimeError("No valid sensor readings")
            
        # Robust statistics: median + MAD filtering
        median = statistics.median(vals)
        if len(vals) == 1:
            return vals[0]
            
        mad = statistics.median([abs(v - median) for v in vals])
        if mad == 0:
            return statistics.mean(vals)
            
        # Filter outliers using 3*MAD rule
        filtered = [v for v in vals if abs(v - median) <= 3.0 * mad]
        return statistics.mean(filtered if filtered else vals)
    
    def start_reading(self):
        """Start the reading thread"""
        thread = threading.Thread(target=self._reading_loop, daemon=True)
        thread.start()
        return thread
    
    def _reading_loop(self):
        """Main reading loop with warmup"""
        self.hx.reset()
        
        # Warmup period
        warmup_end = time.time() + self.config.warmup_sec
        while time.time() < warmup_end and not self.stop_reading.is_set():
            try:
                raw = self._robust_read()
                weight = self.calibration.predict_weight(raw)
                self.ema.update(weight)
                with self.lock:
                    self.history.append((time.time(), self.ema.value))
            except Exception as e:
                self.logger.warning(f"Warmup read error: {e}")
        
        self.baseline_ready.set()
        self.logger.info("Weight reader ready")
        
        # Main reading loop
        read_interval = 1.0 / self.config.read_hz
        while not self.stop_reading.is_set():
            try:
                raw = self._robust_read()
                weight = self.calibration.predict_weight(raw)
                
                with self.lock:
                    self.current_weight = self.ema.update(weight)
                    now = time.time()
                    self.history.append((now, self.current_weight))
                    
                    # Trim old history
                    while self.history and (now - self.history[0][0]) > 10.0:
                        self.history.popleft()
                        
            except Exception as e:
                self.logger.error(f"Reading error: {e}")
                
            time.sleep(read_interval)
    
    def get_current_weight(self) -> float:
        """Thread-safe weight getter"""
        with self.lock:
            return self.current_weight
    
    def get_baseline_snapshot(self, duration: float = None) -> float:
        """Get baseline weight over specified duration"""
        duration = duration or self.config.snapshot_sec
        weights = []
        end_time = time.time() + duration
        
        while time.time() < end_time:
            weights.append(self.get_current_weight())
            time.sleep(0.02)
            
        return statistics.mean(weights) if weights else 0.0
    
    def is_stable(self, window_sec: float = None, band_kg: float = None) -> bool:
        """Check if weight is stable over time window"""
        window_sec = window_sec or self.config.stability_window_sec
        band_kg = band_kg or self.config.stability_band_kg
        
        with self.lock:
            now = time.time()
            recent_weights = [w for t, w in self.history if (now - t) <= window_sec]
            
        if len(recent_weights) < 2:
            return False
            
        return (max(recent_weights) - min(recent_weights)) < band_kg
    
    def close_to_baseline(self, baseline: float, epsilon: float = None) -> bool:
        """Check if current weight is close to baseline"""
        epsilon = epsilon or self.config.baseline_epsilon_kg
        return abs(self.get_current_weight() - baseline) <= epsilon
    
    def stop(self):
        """Stop reading thread"""
        self.stop_reading.set()

# ---------------- Enhanced EMA ----------------
class EMA:
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False
    
    def update(self, x: float) -> float:
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

# ---------------- Video System ----------------
class VideoSystem:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.window_name = "WeightSystem"
        self.screen_size = None
        self.logger = logging.getLogger(__name__)
        self._setup_display()
    
    def _setup_display(self):
        """Setup fullscreen display"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    def _letterbox_frame(self, frame: np.ndarray) -> np.ndarray:
        """Fit frame to screen with letterboxing"""
        if self.screen_size is None:
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)
            _, _, w, h = cv2.getWindowImageRect(self.window_name)
            self.screen_size = (w, h)
        
        screen_w, screen_h = self.screen_size
        frame_h, frame_w = frame.shape[:2]
        
        # Calculate scale to fit
        scale = min(screen_w / frame_w, screen_h / frame_h)
        new_w = max(1, int(frame_w * scale))
        new_h = max(1, int(frame_h * scale))
        
        # Resize and center
        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        
        y_offset = (screen_h - new_h) // 2
        x_offset = (screen_w - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return canvas
    
    def display_frame_with_osd(self, frame: np.ndarray, weight_data: dict) -> bool:
        """Display frame with weight overlay"""
        canvas = self._letterbox_frame(frame)
        
        # Add weight information
        baseline = weight_data.get('baseline', 0.0)
        current = weight_data.get('current', 0.0)
        delta = current - baseline
        state = weight_data.get('state', '')
        extra_info = weight_data.get('extra', '')
        
        # Main weight info
        weight_text = f"Baseline: {baseline:.2f}kg | Current: {current:.2f}kg | Delta: {delta:.2f}kg"
        cv2.putText(canvas, weight_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # State info
        if state:
            cv2.putText(canvas, f"State: {state}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        
        # Extra info
        if extra_info:
            cv2.putText(canvas, extra_info, (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)
        
        cv2.imshow(self.window_name, canvas)
        
        # Check for exit keys
        key = cv2.waitKey(int(1000 / self.config.display_fps)) & 0xFF
        return key in {27, ord('q')}  # ESC or 'q'
    
    def show_error_message(self, message: str, duration: float = 2.0):
        """Show error message for specified duration"""
        error_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        frames_to_show = int(duration * self.config.display_fps)
        for _ in range(frames_to_show):
            canvas = self._letterbox_frame(error_frame)
            cv2.putText(canvas, f"ERROR: {message}", (50, 360), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow(self.window_name, canvas)
            cv2.waitKey(int(1000 / self.config.display_fps))
    
    def cleanup(self):
        """Cleanup video resources"""
        cv2.destroyAllWindows()

# ---------------- Main Application ----------------
class WeightMonitoringSystem:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = setup_logging()
        self.calibration = WeightCalibration(
            os.path.join(config.base_dir, config.calibration_file)
        )
        self.weight_reader = WeightReader(config, self.calibration)
        self.video_system = VideoSystem(config)
        self.current_state = SequenceState.IDLE
        
    def run(self):
        """Main system loop"""
        try:
            self._check_video_files()
            
            # Start weight reading
            reader_thread = self.weight_reader.start_reading()
            self.weight_reader.baseline_ready.wait()
            self.logger.info("System ready - starting main loop")
            
            # Main state machine
            while self.current_state != SequenceState.ABORT:
                if self.current_state == SequenceState.IDLE:
                    self.current_state = self._run_idle_sequence()
                elif self.current_state == SequenceState.MEASURING:
                    self.current_state = self._run_measuring_sequence()
                elif self.current_state == SequenceState.CONFIRMED:
                    self.current_state = self._run_confirmed_sequence()
                    
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"System error: {e}")
        finally:
            self._cleanup()
    
    def _check_video_files(self):
        """Verify all video files exist"""
        missing_files = []
        for name, path in self.config.video_files.items():
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            raise FileNotFoundError(f"Missing video files: {missing_files}")
            
        self.logger.info("All video files found")
    
    def _run_idle_sequence(self) -> SequenceState:
        """Run idle sequence until weight increase detected"""
        video_path = self.config.video_files['idle']
        
        while True:
            baseline = self.weight_reader.get_baseline_snapshot()
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.video_system.show_error_message(f"Cannot open {video_path}")
                return SequenceState.ABORT
            
            weight_increased = False
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_weight = self.weight_reader.get_current_weight()
                delta = current_weight - baseline
                
                weight_data = {
                    'baseline': baseline,
                    'current': current_weight,
                    'state': 'IDLE - Waiting for weight',
                    'extra': f'Threshold: {self.config.trigger_threshold_kg}kg'
                }
                
                if self.video_system.display_frame_with_osd(frame, weight_data):
                    cap.release()
                    return SequenceState.ABORT
                
                if delta >= self.config.trigger_threshold_kg:
                    weight_increased = True
            
            cap.release()
            
            if weight_increased:
                self.logger.info(f"Weight increase detected: {delta:.2f}kg")
                return SequenceState.MEASURING
    
    def _run_measuring_sequence(self) -> SequenceState:
        """Run measuring sequence until stable"""
        baseline = self.weight_reader.get_baseline_snapshot()
        video_path = self.config.video_files['measuring']
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.video_system.show_error_message(f"Cannot open {video_path}")
            return SequenceState.ABORT
        
        stable_detected = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_weight = self.weight_reader.get_current_weight()
            is_stable = self.weight_reader.is_stable()
            
            weight_data = {
                'baseline': baseline,
                'current': current_weight,
                'state': 'MEASURING - Waiting for stability',
                'extra': f'Stable: {is_stable} | Window: {self.config.stability_window_sec}s'
            }
            
            if self.video_system.display_frame_with_osd(frame, weight_data):
                cap.release()
                return SequenceState.ABORT
            
            if is_stable:
                stable_detected = True
        
        cap.release()
        
        if stable_detected:
            if self.weight_reader.close_to_baseline(baseline):
                self.logger.info("Weight returned to baseline")
                return SequenceState.IDLE
            else:
                self.logger.info(f"Stable weight confirmed: {current_weight:.2f}kg")
                return SequenceState.CONFIRMED
        else:
            return SequenceState.MEASURING  # Continue measuring
    
    def _run_confirmed_sequence(self) -> SequenceState:
        """Run confirmation sequence"""
        baseline = self.weight_reader.get_baseline_snapshot()
        current_weight = self.weight_reader.get_current_weight()
        
        # Select appropriate video based on weight
        if 5.0 <= current_weight < 60.0:
            video_path = self.config.video_files['confirmed_normal']
        else:
            video_path = self.config.video_files['confirmed_heavy']
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.video_system.show_error_message(f"Cannot open {video_path}")
            return SequenceState.ABORT
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        pause_frame = int(self.config.seq3_pause_time_sec * fps)
        is_paused = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_weight = self.weight_reader.get_current_weight()
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Check if we should pause
            if not is_paused and frame_number >= pause_frame:
                is_paused = True
                
                # Wait until weight returns to baseline
                while not self.weight_reader.close_to_baseline(baseline):
                    current_weight = self.weight_reader.get_current_weight()
                    
                    weight_data = {
                        'baseline': baseline,
                        'current': current_weight,
                        'state': 'CONFIRMED - Paused',
                        'extra': 'Remove weight to continue...'
                    }
                    
                    if self.video_system.display_frame_with_osd(frame, weight_data):
                        cap.release()
                        return SequenceState.ABORT
            
            weight_data = {
                'baseline': baseline,
                'current': current_weight,
                'state': 'CONFIRMED - Playing',
                'extra': f'Weight confirmed: {current_weight:.2f}kg'
            }
            
            if self.video_system.display_frame_with_osd(frame, weight_data):
                cap.release()
                return SequenceState.ABORT
        
        cap.release()
        self.logger.info("Confirmation sequence complete")
        return SequenceState.IDLE
    
    def _cleanup(self):
        """Cleanup system resources"""
        self.logger.info("Shutting down system...")
        self.weight_reader.stop()
        self.video_system.cleanup()
        GPIO.cleanup()

# ---------------- Entry Point ----------------
if __name__ == "__main__":
    config = SystemConfig()
    system = WeightMonitoringSystem(config)
    system.run()
