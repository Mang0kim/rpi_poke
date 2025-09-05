#!/usr/bin/env python3
import os, time, json, threading, socket, subprocess, atexit
from HX711 import *

# ==================== 경로/상수 ====================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VID_DIR    = os.path.join(BASE_DIR, "vid")
CALIB_PATH = os.path.join(BASE_DIR, "hx711_calibration.json")

THRESH_KG = 1.0
SEQ1_HOLD_SEC   = 1.5
SEQ2_STABLE_SEC = 2.5
SEQ3_FREEZE_SEC = 1.5
SAMPLES = 10
WARMUP_SEC = 5.0

MPV_SPEED = "1.0"
MPV_SOCK  = "/tmp/mpv-smartscale.sock"

# ==================== 전역 ====================
weight_kg = 0.0
running   = True
tare_offset = 0.0
tare_window_s = 1.0

# ==================== 유틸: 파일 선택 ====================
def get_video_path(stem, subdir=None):
    base = os.path.join(VID_DIR, subdir) if subdir else VID_DIR
    cand_fix = os.path.join(base, f"{stem}_fix.mp4")
    cand     = os.path.join(base, f"{stem}.mp4")
    return cand_fix if os.path.isfile(cand_fix) else cand

# ==================== MPV IPC 컨트롤러 (reconnect 내구성 강화) ====================
class MpvIPC:
    def __init__(self, socket_path=MPV_SOCK):
        self.socket_path = socket_path
        self.proc = None
        self.sock = None
        self._launch()
        atexit.register(self.close)

    def _launch(self):
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        except Exception:
            pass
        self.proc = subprocess.Popen([
            "mpv",
            "--idle=yes","--keep-open=yes","--force-window=yes","--fs",
            "--no-osc","--really-quiet",
            f"--speed={MPV_SPEED}",
            f"--input-ipc-server={self.socket_path}",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time()
        while time.time() - t0 < 8.0:
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
        try:
            if self.sock: self.sock.close()
        except Exception:
            pass
        self.sock = None
        if not self._alive():
            try:
                if self.proc: self.proc.terminate()
            except Exception:
                pass
            self.proc = None
            self._launch()
            return
        t0 = time.time()
        while time.time() - t0 < 3.0:
            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.settimeout(0.6)
                self.sock.connect(self.socket_path)
                return
            except Exception:
                time.sleep(0.05)
        try:
            if self.proc: self.proc.terminate()
        except Exception:
            pass
        self.proc = None
        self._launch()

    def _send(self, obj):
        payload = (json.dumps(obj) + "\n").encode("utf-8")
        for _ in (1,2):
            try:
                self.sock.sendall(payload)
                try:
                    _ = self.sock.recv(4096)
                except socket.timeout:
                    pass
                return
            except (BrokenPipeError, OSError):
                self._reconnect()
        raise

    def command(self, *cmd):
        self._send({"command": list(cmd)})

    def set(self, prop, val):
        self.command("set", prop, val)

    def get(self, prop):
        query = (json.dumps({"command": ["get_property", prop]}) + "\n").encode()
        for _ in (1,2):
            try:
                self.sock.sendall(query)
                end = time.time() + 0.6
                buf = b""
                while time.time() < end:
                    try:
                        part = self.sock.recv(4096)
                        if not part: break
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
            if eof: return
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
            if self._alive(): self.command("quit")
        except Exception:
            pass
        try:
            if self.sock: self.sock.close()
        except Exception:
            pass
        try:
            if self.proc and self._alive(): self.proc.terminate()
        except Exception:
            pass
        try:
            if os.path.exists(self.socket_path): os.unlink(self.socket_path)
        except Exception:
            pass

# ==================== 캘리브/측정 ====================
def load_calib():
    with open(CALIB_PATH, "r") as f:
        return json.load(f)

def robust_read_kg(hx, seconds=1.0, group_size=4, gap_s=0.02):
    """빠르게 읽어서 묶음평균 → 중앙값."""
    buf, grp = [], []
    t_end = time.time() + seconds
    while time.time() < t_end:
        try:
            w = float(hx.weight(1))   # 내부 단위 그대로 읽음
            grp.append(w)
            if len(grp) >= group_size:
                buf.append(sum(grp)/len(grp)); grp = []
        except Exception:
            pass
        time.sleep(gap_s)
    if grp: buf.append(sum(grp)/len(grp))
    if not buf: return float('nan')
    buf.sort()
    mid = len(buf)//2
    return buf[mid] if len(buf)%2==1 else 0.5*(buf[mid-1]+buf[mid])

def weight_reader():
    """1초 주기로 a,b만 적용(※ piecewise 미사용)."""
    global weight_kg, running
    cfg = load_calib()
    a, b = cfg["a"], cfg["b"]

    with SimpleHX711(cfg["data_pin"], cfg["clk_pin"], 1, 0) as hx:
        hx.setUnit(Mass.Unit.KG)  # 필요 시 원시카운트 모드로 바꿔도 됨

        print(f"[CAL] linear only: a={a:.8g}, b={b:.8g}")

        while running:
            t0 = time.time()
            try:
                measured = robust_read_kg(hx, seconds=1.0, group_size=4, gap_s=0.02)
                # === piecewise 제거: 항상 a,b만 적용 ===
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
        acc += weight_kg; cnt += 1
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

# ==================== FSM ====================
def main():
    global running
    if not os.path.isdir(VID_DIR):
        print("[ERR] Video folder not found:", VID_DIR)
        return

    mpv = MpvIPC()
    print("[SYS] mpv started (single window).")

    t = threading.Thread(target=weight_reader, daemon=True)
    t.start()

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
            # -------- SEQ1 --------
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

            # -------- SEQ2 --------
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
                            result_bin = kg_bin_floor_for_result(eff)
                            print(f"[SEQ2] result_bin (floor) = {result_bin:02d}")
                            state = 3
                            break
                    time.sleep(0.05)

            # -------- SEQ3 --------
            while state == 3:
                stem = f"ScaleCustom_txt_{result_bin:02d}"
                path = get_video_path(stem, subdir="txt")
                print(f"[SEQ3] enter: result -> {stem}, freeze at {SEQ3_FREEZE_SEC}s")

                mpv.loadfile(path, pause=False, loop_file=False)
                mpv.freeze_at(SEQ3_FREEZE_SEC)

                eff_at_freeze = effective_weight()
                resume_threshold = max(0.0, eff_at_freeze * 0.2)
                print(f"[SEQ3] paused at {SEQ3_FREEZE_SEC}s. eff_at_freeze={eff_at_freeze:.3f}kg, "
                      f"resume_threshold={resume_threshold:.3f}kg")

                while True:
                    eff_now = effective_weight()
                    if eff_now <= resume_threshold:
                        print(f"[SEQ3] drop >=80% detected (eff_now={eff_now:.3f}kg). Resume to end.")
                        break
                    time.sleep(0.1)

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
