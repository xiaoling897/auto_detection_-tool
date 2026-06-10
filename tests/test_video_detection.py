"""视频识别集成测试：用真实 GUI 的 _camera_loop + _inference_worker 跑 data/video 的视频。

验证：① 能按视频帧率读帧并显示（不报错）；② 后台推理能识别出工具；③ 跑一段后能正常停。
不弹窗（root.withdraw），跑约 RUN_SECONDS 秒后主动停。

跑法：
    python tests/test_video_detection.py
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
RUN_SECONDS = 5.0


def main():
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        print(f"✗ {VIDEO_DIR} 下没有 mp4")
        return
    video = videos[0]
    print(f"测试视频: {video.name}")

    root = tk.Tk()
    root.withdraw()                 # 不显示窗口
    app = ToolDetectionApp(root)
    print(f"后端: {'YOLO' if app.use_yolo else 'SIFT'}  工具数: {len(app.tools)}\n")

    # 模拟 open_video 的准备（跳过文件对话框）
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened(), "视频打不开"
    app.cap = cap
    app.use_camera = True
    app.is_video_file = True
    fps = cap.get(cv2.CAP_PROP_FPS)
    app.video_fps = fps if fps and fps > 1 else 25.0
    print(f"视频原生 fps: {app.video_fps:.1f}")
    app.running = True
    app.should_speak = False        # 测试不播报

    # RUN_SECONDS 秒后主动停
    threading.Timer(RUN_SECONDS, lambda: setattr(app, "running", False)).start()

    t0 = time.time()
    app._camera_loop()              # 主线程跑显示循环（内部会起推理线程）
    dur = time.time() - t0

    # 收集结果
    now = time.time()
    recent = sorted(n for n, t in app.last_seen.items()
                    if now - t < max(VISIBILITY_WINDOW, RUN_SECONDS))
    print(f"\n--- 跑了 {dur:.1f}s ---")
    print(f"识别到的工具（{len(recent)}）: {recent or '(无)'}")
    if app.last_counts:
        top = sorted(app.last_counts.items(), key=lambda kv: -kv[1])[:6]
        print("置信度: " + "  ".join(f"{n}={c}%" for n, c in top))

    ok = len(recent) > 0
    print(f"\n{'✓ 通过：视频能跑且识别出工具' if ok else '✗ 没识别到工具（检查视频内容/模型）'}")
    root.destroy()


if __name__ == "__main__":
    main()
