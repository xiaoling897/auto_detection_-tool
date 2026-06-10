"""验证：开启语音播报后，显示线程帧率不再周期性暴跌（语音已移到后台线程）。

跑约 RUN_SECONDS 秒（覆盖至少一次语音播报触发），统计期间显示线程的 fps，
报告最小值——修复前会看到周期性掉到 2~12 fps，修复后应稳定在高位。
"""
import re
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.gui import ToolDetectionApp, SPEAK_INTERVAL  # noqa: E402

VIDEO_DIR = ROOT / "data" / "video" / "视频"
DIAG = ROOT / "data" / "run_diag.txt"
RUN_SECONDS = max(12.0, SPEAK_INTERVAL + 4)


def main():
    video = sorted(VIDEO_DIR.glob("*.mp4"))[0]
    root = tk.Tk()
    root.withdraw()
    app = ToolDetectionApp(root)

    cap = cv2.VideoCapture(str(video))
    app.cap = cap
    app.use_camera = True
    app.is_video_file = True
    fps = cap.get(cv2.CAP_PROP_FPS)
    app.video_fps = fps if fps and fps > 1 else 25.0
    app.running = True
    app.should_speak = True          # 关键：开启语音，复现之前的卡顿场景

    start_lines = len(DIAG.read_text(encoding="utf-8", errors="replace").splitlines())
    threading.Timer(RUN_SECONDS, lambda: setattr(app, "running", False)).start()
    app._camera_loop()

    # 解析本次运行期间写入的"显示线程：X fps"
    lines = DIAG.read_text(encoding="utf-8", errors="replace").splitlines()[start_lines:]
    fps_vals = [float(m.group(1)) for ln in lines
                if (m := re.search(r"显示线程：([\d.]+) fps", ln))]
    print(f"\n语音间隔 {SPEAK_INTERVAL:.0f}s，跑了 {RUN_SECONDS:.0f}s")
    print(f"显示 fps 采样: {[round(v) for v in fps_vals]}")
    if fps_vals:
        lo = min(fps_vals)
        print(f"最低 {lo:.1f} fps，平均 {sum(fps_vals)/len(fps_vals):.1f} fps")
        print("✓ 通过：无周期性暴跌" if lo >= 20
              else f"✗ 仍有掉帧（最低 {lo:.1f}）—— 显示线程仍被阻塞")
    root.destroy()


if __name__ == "__main__":
    main()
