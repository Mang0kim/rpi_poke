#!/usr/bin/env python3
# lightweight_weight_system.py
import os, cv2, time, json, statistics, threading
import numpy as np
from collections import deque
from hx711 import HX711
import RPi.GPIO as GPIO

# ---------------- 설정 ----------------
BASE = os.path.dirname(os.path.abspath(__file__))
VID01 = os.path.join(BASE, "01.mp4")
VID02 = os.path.join(BASE, "02.mp4") 
VID03 = os.path.join(BASE, "03.mp4")
VID03H = os.path.join(BASE, "03-1.mp4")
CALIB = os.path.join(BASE, "calibration.json")

# 성능 최적화된 설정
THRESH_KG = 0.8
STABLE_WINDOW_SEC = 3.0
STABLE_BAND_KG = 1.5
BASELINE_EPS_KG = 0.1
SEQ3_PAUSE_TIME_SEC = 2.0

# 빠른 센서 읽기 설정
EMA_ALPHA = 0.3        # 더 빠른 반응
READS = 3              # 샘플 수 줄임 (10 → 3)
READ_DELAY = 0.001     # 읽기 간격 줄임
READ_HZ = 25.0         # 업데이트 주기 증가 (15 → 25)
WARMUP_SEC = 0.5       # 워밍업 시간 줄임
SNAPSHOT_SEC = 0.4     # 스냅샷 시간 줄임

DISPLAY_FPS = 30
WIN_NAME = "Video"
ESC_KEYS = {27, ord('q')}

# ---------------- 간단한 캘리브레이션 ----------------
def load_calibration(path):
    with open(path, "r") as f:
        d = json.load(f)
    model = d.get("model", "linear")
    coeffs = d["coeffs"] 
    polarity = float(d.get("polarity", 1.0))
    return model, coeffs, polarity

def predict_weight(model, coeffs, polarity, raw):
    x = polarity * raw
    if model == "linear":
        return coeffs["a1"] * x + coeffs["b1"]
    else:
        return coeffs["a2"] * x * x + coeffs["b2"] * x + coeffs["c2"]

# ---------------- 경량 센서 읽기 ----------------
DT_PIN, SCK_PIN, GAIN = 5, 6, 128
hx = HX711(dout_pin=DT_PIN, pd_sck_pin=SCK_PIN, channel='A', gain=GAIN)

def read_fast(n=READS):
    """빠른 센서 읽기 - 복잡한 통계 처리 제거"""
    vals = []
    for _ in range(n):
        try:
            raw = hx.get_raw_data()
            if isinstance(raw, (int, float)):
                vals.append(float(raw))
        except:
            pass
        if READ_DELAY > 0:
            time.sleep(READ_DELAY)
    
    if not vals:
        raise RuntimeError("No sensor data.")
    
    # 간단한 필터링 - 평균만 사용
    return sum(vals) / len(vals)

class FastEMA:
    """간소화된 EMA"""
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.value = 0.0
        self.init = False
    
    def update(self, x):
        if not self.init:
            self.value = x
            self.init = True
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

# ---------------- 전역 변수 (간단함) ----------------
current_weight_kg = 0.0
baseline_ready = threading.Event()
stop_reader = threading.Event()
weight_history = deque(maxlen=200)  # 크기 줄임

def reader_thread(model, coeffs, polarity):
    """경량화된 읽기 스레드"""
    global current_weight_kg
    ema = FastEMA(EMA_ALPHA)
    
    # 워밍업
    t0 = time.time()
    while (time.time() - t0) < WARMUP_SEC and not stop_reader.is_set():
        try:
            raw = read_fast()
            w = predict_weight(model, coeffs, polarity, raw)
            ema.update(w)
        except:
            pass
    
    baseline_ready.set()
    
    # 메인 루프
    while not stop_reader.is_set():
        try:
            raw = read_fast()
            w = predict_weight(model, coeffs, polarity, raw)
            current_weight_kg = ema.update(w)
            
            # 히스토리 관리 - 최소한만
            now = time.time()
            weight_history.append((now, current_weight_kg))
            
            # 오래된 데이터 제거 (4초만 유지)
            while weight_history and (now - weight_history[0][0]) > 4.0:
                weight_history.popleft()
                
        except:
            pass
        
        time.sleep(1.0 / READ_HZ)

# ---------------- 간단한 비디오 처리 ----------------
_screen_size = None

def setup_display():
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def show_frame(frame, baseline, current, extra=""):
    """간소화된 화면 표시"""
    global _screen_size
    
    # 화면 크기 캐싱
    if _screen_size is None:
        cv2.imshow(WIN_NAME, frame)
        cv2.waitKey(1)
        _, _, w, h = cv2.getWindowImageRect(WIN_NAME)
        _screen_size = (w, h)
    
    # 단순 리사이즈 (letterbox 제거)
    screen_w, screen_h = _screen_size
    frame_resized = cv2.resize(frame, (screen_w, screen_h))
    
    # 필수 정보만 표시
    delta = current - baseline
    text = f"B:{baseline:.1f} C:{current:.1f} D:{delta:.1f}"
    cv2.putText(frame_resized, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    
    if extra:
        cv2.putText(frame_resized, extra, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)
    
    cv2.imshow(WIN_NAME, frame_resized)
    key = cv2.waitKey(int(1000/DISPLAY_FPS)) & 0xFF
    return key in ESC_KEYS

# ---------------- 빠른 무게 체크 함수들 ----------------
def get_baseline_snapshot():
    """빠른 베이스라인 측정"""
    weights = []
    end_time = time.time() + SNAPSHOT_SEC
    while time.time() < end_time:
        weights.append(current_weight_kg)
        time.sleep(0.02)
    return sum(weights) / len(weights) if weights else 0.0

def is_stable():
    """간단한 안정성 검사"""
    if len(weight_history) < 10:  # 최소 데이터 필요
        return False
    
    now = time.time()
    recent_weights = [w for t, w in weight_history if (now - t) <= STABLE_WINDOW_SEC]
    
    if len(recent_weights) < 2:
        return False
    
    return (max(recent_weights) - min(recent_weights)) < STABLE_BAND_KG

def close_to_baseline(baseline):
    """베이스라인 근접 검사"""
    return abs(current_weight_kg - baseline) <= BASELINE_EPS_KG

# ---------------- 시퀀스들 ----------------
def seq1_idle():
    """SEQ1: 대기"""
    if not os.path.exists(VID01):
        print("01.mp4 not found")
        return "abort"
    
    while True:
        baseline = get_baseline_snapshot()
        
        cap = cv2.VideoCapture(VID01)
        if not cap.isOpened():
            return "abort"
        
        triggered = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            delta = current_weight_kg - baseline
            
            if show_frame(frame, baseline, current_weight_kg, f"SEQ1 T:{THRESH_KG}"):
                cap.release()
                return "abort"
            
            if delta >= THRESH_KG:
                triggered = True
        
        cap.release()
        
        if triggered:
            return "to_seq2"

def seq2_measuring(baseline):
    """SEQ2: 측정"""
    if not os.path.exists(VID02):
        print("02.mp4 not found") 
        return "abort"
    
    cap = cv2.VideoCapture(VID02)
    if not cap.isOpened():
        return "abort"
    
    stable_found = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        stable = is_stable()
        extra = f"SEQ2 Stable:{stable}"
        
        if show_frame(frame, baseline, current_weight_kg, extra):
            cap.release()
            return "abort"
        
        if stable:
            stable_found = True
    
    cap.release()
    
    if close_to_baseline(baseline):
        return "to_seq1"
    elif stable_found:
        return "to_seq3"
    else:
        return "to_seq2"  # 다시 측정

def seq3_confirmed(baseline):
    """SEQ3: 확인"""
    weight = current_weight_kg
    video_file = VID03 if (5.0 <= weight < 60.0) else VID03H
    
    if not os.path.exists(video_file):
        print(f"{os.path.basename(video_file)} not found")
        return "abort"
    
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return "abort"
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pause_frame = int(SEQ3_PAUSE_TIME_SEC * fps)
    paused = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # 일시정지 처리
        if not paused and frame_num >= pause_frame:
            paused = True
            while not close_to_baseline(baseline):
                if show_frame(frame, baseline, current_weight_kg, "SEQ3 PAUSED"):
                    cap.release()
                    return "abort"
        
        if show_frame(frame, baseline, current_weight_kg, "SEQ3"):
            cap.release()
            return "abort"
    
    cap.release()
    return "to_seq1"

# ---------------- 메인 ----------------
if __name__ == "__main__":
    try:
        print("Starting lightweight weight system...")
        
        setup_display()
        model, coeffs, polarity = load_calibration(CALIB)
        hx.reset()
        
        # 읽기 스레드 시작
        t = threading.Thread(target=reader_thread, args=(model, coeffs, polarity), daemon=True)
        t.start()
        baseline_ready.wait()
        
        print("System ready!")
        
        # 상태 머신
        while True:
            result = seq1_idle()
            if result == "abort":
                break
            
            baseline = get_baseline_snapshot()
            
            result = seq2_measuring(baseline)
            if result == "abort":
                break
            elif result == "to_seq1":
                continue
            
            result = seq3_confirmed(baseline)
            if result == "abort":
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        stop_reader.set()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        print("System stopped")
