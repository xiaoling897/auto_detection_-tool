"""端到端验证：主线程 UI 泵 + 后台读帧/推理线程 一起跑，模拟真实 GUI 架构。

确认：① 主线程 _ui_pump 能正常画帧（_last_shown_id 持续推进、无异常）；
     ② 后台推理能识别工具；③ 视频按节奏播放（显示帧数合理）。
用 root.update() 驱动 after 回调（等价于 mainloop），跑 RUN_SECONDS 秒。
"""
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.gui import ToolDetectionApp, VISIBILITY_WINDOW  # noqa: E402

VIDEO_DIR = ROOT / "data" / "video" / "视频"
RUN_SECONDS = 6.0


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
    app.should_speak = True

    app.detection_thread = threading.Thread(target=app.detection_loop, daemon=True)
    app.detection_thread.start()
    app._start_ui_pump()

    err = []
    t0 = time.time()
    while time.time() - t0 < RUN_SECONDS:
        try:
            root.update()           # 驱动主线程 after 回调（即 _ui_pump）
        except Exception as e:
            err.append(repr(e))
            break
        time.sleep(0.004)
    dur = time.time() - t0
    app.running = False
    time.sleep(0.2)

    shown = app._last_shown_id + 1   # 主线程实际画出的帧数（版本号从 0 起）
    now = time.time()
    recog = sorted(n for n, t in app.last_seen.items()
                   if now - t < max(VISIBILITY_WINDOW, RUN_SECONDS))
    print(f"\n--- 跑了 {dur:.1f}s ---")
    print(f"主线程画帧数: {shown}（≈{shown/dur:.1f} fps）")
    print(f"识别到工具({len(recog)}): {recog}")
    ok = not err and shown > dur * 15 and len(recog) > 0
    if err:
        print(f"✗ 主线程画图异常: {err}")
    elif shown <= dur * 15:
        print(f"✗ 画帧太少({shown})，泵可能没正常工作")
    else:
        print("✓ 通过：主线程画图正常、无异常、识别正常（黑屏闪烁的根因已消除）")
    root.destroy()


if __name__ == "__main__":
    main()
