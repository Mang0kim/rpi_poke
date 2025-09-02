#!/usr/bin/env python3
import os, time, json, threading, socket, subprocess, atexit
from HX711 import *

# -------------------- Paths & Const --------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VID_DIR    = os.path.join(BASE_DIR, "vid")
CALIB_PATH = os.path.join(BASE_DIR, "hx711_calibration.json")

THRESH_KG = 1.0
SEQ1_HOLD_SEC = 1.5      # Seq1 -> Seq2: >=1kg 1.5s 유지
SEQ2_STABLE_SEC = 3.0    # Seq2 안정: round(kg) 3초 동일
SEQ3_FREEZE_SEC = 1.5    # Seq3: 1.5초까지 재생 후 pause
SAMPLES = 10             # hx.weight(10) ≈ 1초
MPV_SPEED = "1.0"        # 항상 1.0배속
MPV_SOCK = "/tmp/mpv-smartscale.sock"

# -------------------- Globals --------------------
weight_kg = 0.0          # a,b 보정 후 실시간 무게
running = True
tare_offset = 0.0        # Seq1에서 영점(평균) 저장
tare_window_s = 1.0

# -------------------- Utility: choose file --------------------
def get_video_path(stem, subdir=None):
    base = os.path.join(VID_DIR, subdir) if subdir else VID_DIR
    cand_fix = os.path.join(base, f"{stem}_fix.mp4")
    cand     = os.path.join(base, f"{stem}.mp4")
    return cand_fix if os.path.isfile(cand_fix) else cand

# -------------------- MPV JSON IPC Controller --------------------
class MpvIPC:
    def __init__(self, socket_path=MPV_SOCK):
        self.socket_path = socket_path
        # stale socket 삭제
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        except Exception:
            pass
        # mpv 실행 (한 번만)
        self.proc = subprocess.Popen([
            "mpv",
            "--idle=yes",
            "--keep-open=yes",
            "--force-window=yes",
            "--fs",
            "--no-osc",
            "--really-quiet",
            f"--speed={MPV_SPEED}",
            f"--input-ipc-server={self.socket_path}"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # 소켓 연결 대기
        t0 = time.time()
        while time.time() - t0 < 5.0:
            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(self.socket_path)
                self.sock.settimeout(0.5)
                break
            except Exception:
                time.sleep(0.05)
        else:
            raise RuntimeError("mpv IPC connect failed")

        atexit.register(self.close)

    def _send(self, obj):
        data = (json.dumps(obj) + "\n").encode("utf-8")
        self.sock.sendall(data)
        # 간단히 한 줄만 읽어서 버퍼 소비 (응답 사용 최소화)
        try:
            self.sock.recv(4096)
        except socket.timeout:
            pass

    def command(self, *cmd):
        self._send({"command": list(cmd)})

    def set(self, prop, val):
        self.command("set", prop, val)

    def get(self, prop):
        # 동기 get (간단 구현): 요청 보내고 짧게 재시도
        self.sock.sendall((json.dumps({"command":["get_property", prop]})+"\n").encode())
        end = time.time() + 0.4
        buf = b""
        while time.time() < end:
            try:
                part = self.sock.recv(4096)
                if not part: break
                buf += part
                # 매우 단순한 파싱 (응답 한 건 가정)
                try:
                    txt = buf.decode(errors="ignore")
                    for line in txt.strip().splitlines():
                        try:
                            obj = json.loads(line)
                            if obj.get("data") is not None:
                                return obj["data"]
                        except Exception:
                            continue
                except Exception:
                    pass
            except socket.timeout:
                break
        return None

    # 고수준 명령
    def loadfile(self, path, pause=False, loop_file=False, start=None, end=None):
        # replace 모드로 로드
        self.command("loadfile", path, "replace")
        self.set("loop-file", "yes" if loop_file else "no")
        if start is not None:
            self.command("seek", str(start), "absolute")
        if end is not None:
            # ab-loop로 끝 시점 고정
            self.set("ab-loop-a", 0)
            self.set("ab-loop-b", float(end))
        else:
            self.set("ab-loop-a", "no")
            self.set("ab-loop-b", "no")
        self.set("pause", "yes" if pause else "no")

    def wait_until_eof(self):
        # duration/ time-pos를 폴링하여 종료까지 블럭
        # keep-open이 켜져 있어도 eof-reached True로 변함
        while True:
            eof = self.get("eof-reached")
            if eof: return
            time.sleep(0.02)

    def freeze_at(self, t_sec):
        # t_sec 지점에 도달하면 pause
        self.set("pause", "no")
        while True:
            pos = self.get("time-pos") or 0.0
            if float(pos) >= float(t_sec):
                self.set("pause", "yes")
                return
            time.sleep(0.01)

    def close(self):
        try:
            self.command("quit")
        except Exception:
            pass
        try:
            if hasattr(self, "sock"):
                self.sock.close()
        except Exception:
            pass
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        except Exception:
            pass

# -------------------- Weight & Tare --------------------
def load_calib():
    with open(CALIB_PATH, "r") as f:
        return json.load(f)

def weight_reader():
    """1초 주기로 무게 갱신 + 프린트"""
    global weight_kg, running
    cfg = load_calib()
    a, b = cfg["a"], cfg["b"]
    with SimpleHX711(cfg["data_pin"], cfg["clk_pin"], 1, 0) as hx:
        hx.setUnit(Mass.Unit.KG)
        while running:
            t0 = time.time()
            try:
                measured = float(hx.weight(SAMPLES))
                weight_kg = a * measured + b
            except Exception:
                pass
            eff = max(0.0, weight_kg - tare_offset)
            print(f"[W] raw={weight_kg:.3f}kg  tare={tare_offset:.3f}kg  eff={eff:.3f}kg")
            dt = time.time() - t0
            time.sleep(max(0, 1.0 - dt))

def effective_weight():
    return max(0.0, weight_kg - tare_offset)

def seq1_update_tare():
    """Seq1 동안 1초 평균으로 영점 갱신"""
    global tare_offset
    acc = 0.0; cnt = 0
    t_begin = time.time()
    while time.time() - t_begin < tare_window_s:
        acc += weight_kg
        cnt += 1
        time.sleep(0.05)
    if cnt > 0:
        tare_offset = acc / cnt
        print(f"[SEQ1] tare updated -> {tare_offset:.3f}kg")

# -------------------- Binning --------------------
def kg_bin_round(x):
    if x < 1.0: return None
    b = int(round(x))
    return max(1, min(100, b))

def kg_bin_floor_for_result(x):
    if x < 1.0: return None
    b = int(x)
    return max(1, min(100, b))

# -------------------- FSM --------------------
def main():
    global running

    if not os.path.isdir(VID_DIR):
        print("[ERR] Video folder not found:", VID_DIR)
        return

    # mpv 1회 실행 (창 유지)
    mpv = MpvIPC()
    print("[SYS] mpv started (single window).")

    # weight thread
    t = threading.Thread(target=weight_reader, daemon=True)
    t.start()

    state = 1
    seq1_hold_start = None
    seq2_bin = None
    seq2_bin_start = None

    try:
        while True:
            # ------------- SEQ1: 대기 (01 반복 재생, 영점 반영) -------------
            while state == 1:
                print("[SEQ1] enter: waiting / playing 01")
                seq1_update_tare()  # 영점 반영

                # 01을 로드하고 loop-file=yes로 끝까지 재생/반복
                mpv.loadfile(get_video_path("01"), pause=False, loop_file=True)
                # loop-file=on이지만 상태 전환은 조건 충족 + "현재 파일 한 사이클 끝" 후로
                # -> 여기서는 간단히 주기적으로 조건 체크 + eof 대기 대신 time-pos reset 감지도 가능
                #   (간소화를 위해 sleep 루프)
                last_cycle_time = time.time()
                while state == 1:
                    eff = effective_weight()
                    if eff >= THRESH_KG:
                        if seq1_hold_start is None:
                            seq1_hold_start = time.time()
                            print("[SEQ1] >=1kg detected; hold timer start")
                        elif time.time() - seq1_hold_start >= SEQ1_HOLD_SEC:
                            print("[SEQ1] hold satisfied (>=1kg for 1.5s). Finishing current 01 cycle...")
                            # 현재 사이클 종료 대기: duration - time-pos 가 작아질 때까지 기다렸다가 eof
                            # (간단화) 잠깐 기다렸다 다음 상태로
                            time.sleep(0.1)
                            # loop를 끄고 끝까지 재생시켜 한 사이클 마무리
                            mpv.set("loop-file", "no")
                            mpv.wait_until_eof()
                            state = 2
                            seq1_hold_start = None
                            seq2_bin = None
                            seq2_bin_start = None
                            print("[SEQ1] -> SEQ2")
                            break
                    else:
                        if seq1_hold_start is not None:
                            print("[SEQ1] hold reset (<1kg)")
                        seq1_hold_start = None
                    time.sleep(0.05)

            # ------------- SEQ2: 측정 (02 반복 재생, bin 3초 고정) -------------
            while state == 2:
                print("[SEQ2] enter: measuring / playing 02")
                mpv.loadfile(get_video_path("02"), pause=False, loop_file=True)

                seq2_bin = None
                seq2_bin_start = None
                while state == 2:
                    eff = effective_weight()
                    now = time.time()

                    if eff < THRESH_KG:
                        print("[SEQ2] dropped <1kg; finishing current 02 cycle and back to SEQ1")
                        mpv.set("loop-file", "no")
                        mpv.wait_until_eof()
                        state = 1
                        break

                    current_bin = kg_bin_round(eff)
                    if current_bin is None:
                        seq2_bin = None; seq2_bin_start = None
                    elif seq2_bin is None or current_bin != seq2_bin:
                        seq2_bin = current_bin
                        seq2_bin_start = now
                        print(f"[SEQ2] bin -> {seq2_bin:02d} (timer reset)")
                    else:
                        held = now - seq2_bin_start
                        if held >= SEQ2_STABLE_SEC:
                            print(f"[SEQ2] bin {seq2_bin:02d} stable for {SEQ2_STABLE_SEC}s. Finish 02 and go SEQ3.")
                            mpv.set("loop-file", "no")
                            mpv.wait_until_eof()
                            result_bin = kg_bin_floor_for_result(eff)  # [N,N+1)
                            print(f"[SEQ2] result_bin (floor) = {result_bin:02d}")
                            state = 3
                            break
                        # 진행상황 로그(optional)
                        # print(f"[SEQ2] holding {seq2_bin:02d}: {held:.1f}/{SEQ2_STABLE_SEC}s")
                    time.sleep(0.05)

           # ------------- SEQ3: 결과 (txt/XX, 1.5s에서 pause -> <1kg 후 나머지 재생) -------------
            while state == 3:
                stem = f"ScaleCustom_txt_{result_bin:02d}"
                path = get_video_path(stem, subdir="txt")
                print(f"[SEQ3] enter: result -> {stem}, freeze at {SEQ3_FREEZE_SEC}s")

                # 1) 영상 로드 후 0부터 재생 시작
                mpv.loadfile(path, pause=False, loop_file=False)

                # 2) 1.5초 지점 도달 시 pause
                mpv.freeze_at(SEQ3_FREEZE_SEC)
                print(f"[SEQ3] paused at {SEQ3_FREEZE_SEC}s. Holding while >=1kg...")

                # 3) 무게가 <1kg 될 때까지 정지 유지
                while effective_weight() >= THRESH_KG:
                    time.sleep(0.1)

                print("[SEQ3] weight <1kg detected. Resume playback to end.")

                # 4) pause 해제하고 끝까지 재생
                mpv.set("pause", "no")
                mpv.wait_until_eof()
                print("[SEQ3] video finished. -> SEQ1")

                state = 1
                break

    except KeyboardInterrupt:
        print("[SYS] KeyboardInterrupt")
    finally:
        running = False
        try:
            mpv.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
