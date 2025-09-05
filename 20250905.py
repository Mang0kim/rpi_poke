#!/usr/bin/env python3
import os, time, json, threading, socket, subprocess, atexit, bisect
from HX711 import *

# ==================== 경로/상수 ====================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VID_DIR    = os.path.join(BASE_DIR, "vid")
CALIB_PATH = os.path.join(BASE_DIR, "hx711_calibration.json")
MODEL_PATH = os.path.join(BASE_DIR, "hx711_model.json")

THRESH_KG = 1.0
SEQ1_HOLD_SEC   = 1.5    # Seq1 -> Seq2: eff ≥ 1kg 1.5s 유지
SEQ2_STABLE_SEC = 2.5    # Seq2: round(kg) 3s 동일
SEQ3_FREEZE_SEC = 1.5    # Seq3: 1.5s 프레임에서 일시정지
SAMPLES = 10             # hx.weight(10) = 약 1s @ 10SPS
WARMUP_SEC = 5.0         # 초기 예열 시간 (영상 재생 없이)

MPV_SPEED = "1.0"        # 항상 1.0배속
MPV_SOCK  = "/tmp/mpv-smartscale.sock"

# ==================== 전역 ====================
weight_kg = 0.0          # 변환 후 실시간 무게
running   = True
tare_offset = 0.0        # Seq1에서 영점(평균) 저장
tare_window_s = 1.0

# ==================== 유틸: 파일 선택 ====================
def get_video_path(stem, subdir=None):
    base = os.path.join(VID_DIR, subdir) if subdir else VID_DIR
    cand_fix = os.path.join(base, f"{stem}_fix.mp4")
    cand     = os.path.join(base, f"{stem}.mp4")
    return cand_fix if os.path.isfile(cand_fix) else cand

# ==================== MPV IPC 컨트롤러 ====================
# ==================== MPV IPC 컨트롤러 (reconnect 내구성 강화) ====================
class MpvIPC:
    def __init__(self, socket_path=MPV_SOCK):
        self.socket_path = socket_path
        self.proc = None
        self.sock = None
        self._launch()
        atexit.register(self.close)

    # --- low-level ---
    def _launch(self):
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        except Exception:
            pass

        self.proc = subprocess.Popen([
            "mpv",
            "--idle=yes",
            "--keep-open=yes",
            "--force-window=yes",
            "--fs",
            "--no-osc",
            "--really-quiet",
            f"--speed={MPV_SPEED}",
            f"--input-ipc-server={self.socket_path}",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 기다렸다가 연결
        t0 = time.time()
        while time.time() - t0 < 8.0:   # 여유 조금 증가
            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.settimeout(0.6)
                self.sock.connect(self.socket_path)
                return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError("mpv IPC connect failed")

    def _alive(self):
        return (self.proc is not None) and (self.proc.poll() is None)

    def _reconnect(self):
        # 소켓 정리
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None

        # mpv가 죽었으면 재실행
        if not self._alive():
            try:
                if self.proc:
                    self.proc.terminate()
            except Exception:
                pass
            self.proc = None
            self._launch()
            return

        # 살아있는데 소켓만 끊겼다면 소켓만 재연결
        t0 = time.time()
        while time.time() - t0 < 3.0:
            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.settimeout(0.6)
                self.sock.connect(self.socket_path)
                return
            except Exception:
                time.sleep(0.05)
        # 안 되면 아예 재실행
        try:
            if self.proc:
                self.proc.terminate()
        except Exception:
            pass
        self.proc = None
        self._launch()

    def _send(self, obj):
        payload = (json.dumps(obj) + "\n").encode("utf-8")
        for attempt in (1, 2):  # 1회 재시도
            try:
                self.sock.sendall(payload)
                try:
                    # 응답 한 번 비워주기 (있을 수도, 없을 수도)
                    _ = self.sock.recv(4096)
                except socket.timeout:
                    pass
                return
            except (BrokenPipeError, OSError):
                # 연결 복구 후 재시도
                self._reconnect()
        # 두 번 실패하면 예외 전파
        raise

    # --- public ---
    def command(self, *cmd):
        self._send({"command": list(cmd)})

    def set(self, prop, val):
        self.command("set", prop, val)

    def get(self, prop):
        # get은 실패 시 자동 복구 후 한 번만 재시도
        query = (json.dumps({"command": ["get_property", prop]}) + "\n").encode()
        for attempt in (1, 2):
            try:
                self.sock.sendall(query)
                end = time.time() + 0.6
                buf = b""
                while time.time() < end:
                    try:
                        part = self.sock.recv(4096)
                        if not part:
                            break
                        buf += part
                        txt = buf.decode(errors="ignore")
                        for line in txt.strip().splitlines():
                            try:
                                obj = json.loads(line)
                                if obj.get("data") is not None:
                                    return obj["data"]
                            except Exception:
                                continue
                    except socket.timeout:
                        break
                return None
            except (BrokenPipeError, OSError):
                self._reconnect()
        return None

    def loadfile(self, path, pause=False, loop_file=False, start=None, end=None):
        self.command("loadfile", path, "replace")
        self.set("loop-file", "yes" if loop_file else "no")
        if start is not None:
            self.command("seek", str(start), "absolute")
        if end is not None:
            self.set("ab-loop-a", 0)
            self.set("ab-loop-b", float(end))
        else:
            self.set("ab-loop-a", "no")
            self.set("ab-loop-b", "no")
        self.set("pause", "yes" if pause else "no")

    def wait_until_eof(self):
        while True:
            eof = self.get("eof-reached")
            if eof:
                return
            time.sleep(0.02)

    def freeze_at(self, t_sec):
        self.set("pause", "no")
        while True:
            pos = self.get("time-pos") or 0.0
            if float(pos) >= float(t_sec):
                self.set("pause", "yes")
                return
            time.sleep(0.01)

    def close(self):
        try:
            if self._alive():
                self.command("quit")
        except Exception:
            pass
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        try:
            if self.proc and self._alive():
                self.proc.terminate()
        except Exception:
            pass
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        except Exception:
            pass

# ==================== 피스와이즈 캘리브 모델 ====================
def load_piecewise_model(path=MODEL_PATH):
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        M = json.load(f)
    if M.get("model") != "piecewise":
        return None
    pts  = M.get("points", [])
    segs = M.get("segments", [])
    ms = [p["measured"] for p in pts]
    return {"ms": ms, "segs": segs, "pts": pts}

def apply_piecewise(M, measured):
    ms, segs = M["ms"], M["segs"]
    if not segs:
        return measured
    i = bisect.bisect_right(ms, measured) - 1
    i = max(0, min(i, len(segs)-1))
    s = segs[i]
    return s["slope"] * measured + s["intercept"]

MODEL = load_piecewise_model()  # 있으면 사용, 없으면 a,b 폴백

def load_calib():
    with open(CALIB_PATH, "r") as f:
        return json.load(f)

def robust_read_kg(hx, seconds=1.0, group_size=4, gap_s=0.02):
    """
    빠르게 단일 샘플을 읽어 group_size개씩 평균 -> 그 평균들의 중앙값 반환.
    - 스파이크/간헐노이즈에 강함.
    """
    import statistics as stats
    buf, grp = [], []
    t_end = time.time() + seconds
    while time.time() < t_end:
        try:
            w = float(hx.weight(1))  # 1샘플씩 빠르게
            grp.append(w)
            if len(grp) >= group_size:
                buf.append(sum(grp)/len(grp))
                grp = []
        except Exception:
            pass
        time.sleep(gap_s)
    if grp:
        buf.append(sum(grp)/len(grp))
    if not buf:
        return float('nan')
    buf.sort()
    mid = len(buf)//2
    return buf[mid] if len(buf)%2==1 else 0.5*(buf[mid-1]+buf[mid])

def weight_reader():
    """1초 주기로 robust 평균 + 모델 적용 후 무게 갱신 & 로그 출력."""
    global weight_kg, running
    cfg = load_calib()
    a, b = cfg["a"], cfg["b"]  # 폴백용

    with SimpleHX711(cfg["data_pin"], cfg["clk_pin"], 1, 0) as hx:
        hx.setUnit(Mass.Unit.KG)

        if MODEL is not None:
            print("[CAL] piecewise model loaded:")
            for p in MODEL["pts"]:
                print(f"      measured~{p['measured']:.4f} -> actual={p['actual']:.4f}")
        else:
            print(f"[CAL] no model; fallback to linear a,b = {a:.6f}, {b:.6f}")

        while running:
            t0 = time.time()
            try:
                measured = robust_read_kg(hx, seconds=1.0, group_size=4, gap_s=0.02)
                if MODEL is not None:
                    weight_kg = apply_piecewise(MODEL, measured)
                else:
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
    """Seq1 동안 1초 평균으로 영점 갱신."""
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

def kg_bin_round(x):
    if x < 1.0: return None
    b = int(round(x))
    return max(1, min(100, b))

def kg_bin_floor_for_result(x):
    if x < 1.0: return None
    b = int(x)
    return max(1, min(100, b))

def main():
    global running

    if not os.path.isdir(VID_DIR):
        print("[ERR] Video folder not found:", VID_DIR)
        return

    # mpv 1회 실행
    mpv = MpvIPC()
    print("[SYS] mpv started (single window).")

    # 측정 스레드 시작
    t = threading.Thread(target=weight_reader, daemon=True)
    t.start()

    # ---------- 초기 예열 ----------
    print(f"[SYS] warm-up for {WARMUP_SEC:.1f} sec ...")
    t_end = time.time() + WARMUP_SEC
    while time.time() < t_end:
        time.sleep(0.1)
    print("[SYS] warm-up finished. Enter SEQ1.")

    state = 1
    seq1_hold_start = None
    seq2_bin = None
    seq2_bin_start = None

    try:
        while True:
            # -------- SEQ1: 대기 (01 반복, 영점 반영) --------
            while state == 1:
                print("[SEQ1] enter: waiting / playing 01")
                seq1_update_tare()

                mpv.loadfile(get_video_path("01"), pause=False, loop_file=True)
                while state == 1:
                    eff = effective_weight()
                    if eff >= THRESH_KG:
                        if seq1_hold_start is None:
                            seq1_hold_start = time.time()
                            print("[SEQ1] >=1kg detected; hold timer start")
                        elif time.time() - seq1_hold_start >= SEQ1_HOLD_SEC:
                            print("[SEQ1] hold satisfied. Finishing current 01 cycle...")
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

            # -------- SEQ2: 측정 (02 반복, round-bin 3s) --------
            while state == 2:
                print("[SEQ2] enter: measuring / playing 02")
                mpv.loadfile(get_video_path("02"), pause=False, loop_file=True)

                seq2_bin = None
                seq2_bin_start = None
                while state == 2:
                    eff = effective_weight()
                    now = time.time()

                    if eff < THRESH_KG:
                        print("[SEQ2] dropped <1kg; finishing 02 -> SEQ1")
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
                            print(f"[SEQ2] bin {seq2_bin:02d} stable for {SEQ2_STABLE_SEC}s. Finish 02 -> SEQ3.")
                            mpv.set("loop-file", "no")
                            mpv.wait_until_eof()
                            result_bin = kg_bin_floor_for_result(eff)  # [N,N+1)
                            print(f"[SEQ2] result_bin (floor) = {result_bin:02d}")
                            state = 3
                            break
                    time.sleep(0.05)

            # -------- SEQ3: 결과 (1.5s에서 pause -> 80% 이상 하중 감소 시 재생 재개) --------
            while state == 3:
                stem = f"ScaleCustom_txt_{result_bin:02d}"
                path = get_video_path(stem, subdir="txt")
                print(f"[SEQ3] enter: result -> {stem}, freeze at {SEQ3_FREEZE_SEC}s")

                # 0부터 재생 시작 → 1.5s에서 일시정지
                mpv.loadfile(path, pause=False, loop_file=False)
                mpv.freeze_at(SEQ3_FREEZE_SEC)

                # freeze 직후 기준 하중 저장
                eff_at_freeze = effective_weight()
                resume_threshold = max(0.0, eff_at_freeze * 0.2)  # 80% 이상 감소 → 20% 이하
                print(f"[SEQ3] paused at {SEQ3_FREEZE_SEC}s. eff_at_freeze={eff_at_freeze:.3f}kg, "
                      f"resume_threshold={resume_threshold:.3f}kg")

                # 일시정지 유지: eff가 기준의 20% 이하가 될 때까지
                while True:
                    eff_now = effective_weight()
                    if eff_now <= resume_threshold:
                        print(f"[SEQ3] drop >=80% detected (eff_now={eff_now:.3f}kg). Resume to end.")
                        break
                    time.sleep(0.1)

                # pause 해제하고 끝까지 재생
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
