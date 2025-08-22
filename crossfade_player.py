#!/usr/bin/env python3
# headless_weight_monitor.py - GUI 문제 발생시 대안
import os, time, json, statistics, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import deque
from hx711 import HX711
import RPi.GPIO as GPIO

# 기본 설정은 동일
BASE = os.path.dirname(os.path.abspath(__file__))
CALIB = os.path.join(BASE, "calibration.json")

# 무게 관련 설정
THRESH_KG = 0.8
STABLE_WINDOW_SEC = 3.0
STABLE_BAND_KG = 1.5
BASELINE_EPS_KG = 0.1

# 센서 설정
EMA_ALPHA = 0.3
READS = 3
READ_HZ = 25.0
DT_PIN, SCK_PIN, GAIN = 5, 6, 128

# 전역 변수
current_weight_kg = 0.0
current_state = "IDLE"
baseline_weight = 0.0
system_status = "Starting..."
weight_history = deque(maxlen=200)
stop_reader = threading.Event()

def load_calibration(path):
    with open(path, "r") as f:
        d = json.load(f)
    return d.get("model", "linear"), d["coeffs"], float(d.get("polarity", 1.0))

def predict_weight(model, coeffs, polarity, raw):
    x = polarity * raw
    if model == "linear":
        return coeffs["a1"] * x + coeffs["b1"]
    else:
        return coeffs["a2"] * x * x + coeffs["b2"] * x + coeffs["c2"]

class FastEMA:
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

def reader_thread(model, coeffs, polarity):
    global current_weight_kg, system_status
    
    hx = HX711(dout_pin=DT_PIN, pd_sck_pin=SCK_PIN, channel='A', gain=GAIN)
    hx.reset()
    ema = FastEMA()
    
    system_status = "Warming up..."
    
    # 워밍업
    for _ in range(20):
        if stop_reader.is_set():
            return
        try:
            vals = []
            for _ in range(READS):
                raw = hx.get_raw_data()
                if isinstance(raw, (int, float)):
                    vals.append(float(raw))
                time.sleep(0.001)
            
            if vals:
                avg_raw = sum(vals) / len(vals)
                weight = predict_weight(model, coeffs, polarity, avg_raw)
                ema.update(weight)
        except:
            pass
        time.sleep(0.05)
    
    system_status = "Ready"
    
    # 메인 루프
    while not stop_reader.is_set():
        try:
            vals = []
            for _ in range(READS):
                raw = hx.get_raw_data()
                if isinstance(raw, (int, float)):
                    vals.append(float(raw))
                time.sleep(0.001)
            
            if vals:
                avg_raw = sum(vals) / len(vals)
                weight = predict_weight(model, coeffs, polarity, avg_raw)
                current_weight_kg = ema.update(weight)
                
                now = time.time()
                weight_history.append((now, current_weight_kg))
                
                # 오래된 데이터 제거
                while weight_history and (now - weight_history[0][0]) > 5.0:
                    weight_history.popleft()
        except:
            pass
        
        time.sleep(1.0 / READ_HZ)

def is_stable():
    if len(weight_history) < 10:
        return False
    
    now = time.time()
    recent_weights = [w for t, w in weight_history if (now - t) <= STABLE_WINDOW_SEC]
    
    if len(recent_weights) < 2:
        return False
    
    return (max(recent_weights) - min(recent_weights)) < STABLE_BAND_KG

def get_baseline_snapshot():
    weights = []
    end_time = time.time() + 0.4
    while time.time() < end_time:
        weights.append(current_weight_kg)
        time.sleep(0.02)
    return sum(weights) / len(weights) if weights else 0.0

# 웹 서버
class WeightHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            delta = current_weight_kg - baseline_weight
            stable = is_stable()
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Weight Monitor</title>
                <meta http-equiv="refresh" content="1">
                <style>
                    body {{ font-family: Arial; margin: 40px; background: #f0f0f0; }}
                    .container {{ background: white; padding: 30px; border-radius: 10px; }}
                    .weight {{ font-size: 3em; color: #333; }}
                    .info {{ font-size: 1.5em; margin: 10px 0; }}
                    .status {{ padding: 10px; border-radius: 5px; }}
                    .idle {{ background: #e3f2fd; }}
                    .measuring {{ background: #f3e5f5; }}
                    .confirmed {{ background: #e8f5e8; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>무게 모니터링 시스템</h1>
                    
                    <div class="weight">현재 무게: {current_weight_kg:.2f} kg</div>
                    
                    <div class="info">베이스라인: {baseline_weight:.2f} kg</div>
                    <div class="info">델타: {delta:.2f} kg</div>
                    <div class="info">안정성: {'안정' if stable else '불안정'}</div>
                    
                    <div class="status {current_state.lower()}">
                        <strong>상태: {current_state}</strong>
                    </div>
                    
                    <div class="info">시스템: {system_status}</div>
                    
                    <p>임계값: {THRESH_KG}kg | 안정 범위: {STABLE_BAND_KG}kg</p>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        
        elif self.path == '/api/weight':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            data = {
                'current': current_weight_kg,
                'baseline': baseline_weight,
                'delta': current_weight_kg - baseline_weight,
                'state': current_state,
                'stable': is_stable(),
                'status': system_status
            }
            
            import json
            self.wfile.write(json.dumps(data).encode())

def state_machine():
    global current_state, baseline_weight
    
    while not stop_reader.is_set():
        if current_state == "IDLE":
            current_state = "IDLE - 대기 중"
            baseline_weight = get_baseline_snapshot()
            
            # 무게 증가 대기
            start_time = time.time()
            while (time.time() - start_time) < 30:  # 30초마다 베이스라인 갱신
                if stop_reader.is_set():
                    return
                
                delta = current_weight_kg - baseline_weight
                if delta >= THRESH_KG:
                    current_state = "MEASURING"
                    break
                time.sleep(0.1)
        
        elif current_state == "MEASURING":
            current_state = "MEASURING - 측정 중"
            baseline_weight = get_baseline_snapshot()
            
            # 안정화 대기
            stable_time = 0
            while stable_time < 3.0:  # 3초간 안정
                if stop_reader.is_set():
                    return
                
                if is_stable():
                    stable_time += 0.1
                else:
                    stable_time = 0
                
                time.sleep(0.1)
            
            # 베이스라인 근처인지 확인
            if abs(current_weight_kg - baseline_weight) <= BASELINE_EPS_KG:
                current_state = "IDLE"
            else:
                current_state = "CONFIRMED"
        
        elif current_state == "CONFIRMED":
            current_state = "CONFIRMED - 확인됨"
            
            # 무게 제거 대기
            while abs(current_weight_kg - baseline_weight) > BASELINE_EPS_KG:
                if stop_reader.is_set():
                    return
                time.sleep(0.1)
            
            current_state = "IDLE"
        
        time.sleep(0.1)

if __name__ == "__main__":
    try:
        print("Starting headless weight monitor...")
        
        model, coeffs, polarity = load_calibration(CALIB)
        
        # 스레드 시작
        reader = threading.Thread(target=reader_thread, args=(model, coeffs, polarity), daemon=True)
        state = threading.Thread(target=state_machine, daemon=True)
        
        reader.start()
        state.start()
        
        # 웹 서버 시작
        server = HTTPServer(('0.0.0.0', 8080), WeightHandler)
        print("Web server running on http://localhost:8080")
        print("Press Ctrl+C to stop")
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stop_reader.set()
        GPIO.cleanup()
        print("System stopped")
