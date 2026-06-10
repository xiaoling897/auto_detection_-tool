"""摄像头流畅度基准：对比"单线程(改前)"与"双线程(改后)"的显示帧率。

用 data/video 下的视频当输入源（比真摄像头快，能更清楚暴露瓶颈）。模拟显示开销
（cvtColor + PIL 缩略图，≈_show_frame 的主要成本），但不依赖 Tk。

跑法：
    python tests/bench_camera_smoothness.py
结论看：双线程的"显示 fps"应远高于单线程——证明视频流畅度已和模型重量脱钩。
"""
import sys
import threading
import time
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.yolo_detector import YoloToolDetector  # noqa: E402

VIDEO_DIR = ROOT / "data" / "video" / "视频"
MODEL = ROOT / "data" / "yolo" / "best.pt"
CLASS_MAP = ROOT / "data" / "yolo" / "class_map.json"
DETECT_INTERVAL = 0.4
RUN_SECONDS = 8.0


def find_video():
    for p in sorted(VIDEO_DIR.glob("*.mp4")):
        return p
    raise FileNotFoundError(f"{VIDEO_DIR} 下没有 mp4")


def read_loop(cap):
    """读一帧，读到结尾就回到开头，永远有帧。"""
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    return frame


def fake_show(frame):
    """模拟 _show_frame 的主要开销（不含 Tk PhotoImage）。"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img.thumbnail((900, 650))


def bench_single_thread(det, video):
    """改前：单线程——读帧→每0.4s整幅 render→显示，一条龙。"""
    cap = cv2.VideoCapture(str(video))
    last_detect = 0.0
    last_boxes = []
    frames = 0
    infers = 0
    t0 = time.time()
    while time.time() - t0 < RUN_SECONDS:
        frame = read_loop(cap)
        now = time.time()
        if now - last_detect >= DETECT_INTERVAL:
            last_detect = now
            _, _, last_boxes = det.infer(frame)   # 推理阻塞本线程
            infers += 1
        disp = det.render(frame, last_boxes) if last_boxes else frame  # 每帧整幅 PIL
        fake_show(disp)
        frames += 1
    cap.release()
    dur = time.time() - t0
    return frames / dur, infers / dur


def bench_two_thread(det, video):
    """改后：双线程——显示线程读帧+render_live(轻量)+显示；推理线程后台跑。"""
    cap = cv2.VideoCapture(str(video))
    state = {"frame": None, "boxes": [], "running": True, "infers": 0}
    lock = threading.Lock()

    def worker():
        last_detect = 0.0
        while state["running"]:
            now = time.time()
            if now - last_detect < DETECT_INTERVAL:
                time.sleep(0.01)
                continue
            with lock:
                f = None if state["frame"] is None else state["frame"].copy()
            if f is None:
                time.sleep(0.01)
                continue
            last_detect = now
            _, _, boxes = det.infer(f)
            state["boxes"] = boxes
            state["infers"] += 1

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    frames = 0
    t0 = time.time()
    while time.time() - t0 < RUN_SECONDS:
        frame = read_loop(cap)
        with lock:
            state["frame"] = frame
        boxes = state["boxes"]
        disp = det.render_live(frame.copy(), boxes) if boxes else frame  # 轻量贴图
        fake_show(disp)
        frames += 1
    state["running"] = False
    th.join(timeout=2)
    cap.release()
    dur = time.time() - t0
    return frames / dur, state["infers"] / dur


def main():
    video = find_video()
    print(f"测试视频: {video.name}")
    print(f"模型: {MODEL.name}\n")
    det = YoloToolDetector(str(MODEL), str(CLASS_MAP))
    # 预热
    cap = cv2.VideoCapture(str(video))
    det.infer(read_loop(cap))
    cap.release()

    print(f"--- 各跑 {RUN_SECONDS:.0f}s ---\n")
    s_disp, s_inf = bench_single_thread(det, video)
    print(f"[改前·单线程] 显示 {s_disp:5.1f} fps   推理 {s_inf:4.1f} 次/s")
    t_disp, t_inf = bench_two_thread(det, video)
    print(f"[改后·双线程] 显示 {t_disp:5.1f} fps   推理 {t_inf:4.1f} 次/s")
    print(f"\n显示帧率提升: {t_disp / max(s_disp, 0.1):.1f}x"
          f"（{s_disp:.1f} → {t_disp:.1f} fps）")


if __name__ == "__main__":
    main()
