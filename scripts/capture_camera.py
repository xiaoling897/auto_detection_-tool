"""从摄像头采集训练图：实时预览，按空格拍一张，按 Q 退出。

用于补充"实际部署摄像头视角"的训练数据（俯拍/较远/光照与手机近拍不同）。
保存到 data/new_shots/cam_XXXX.jpg。

用法：
  1) 先关掉 run.py（释放摄像头）
  2) python scripts/capture_camera.py
  3) 摆好工具 → 按【空格】拍一张 → 换个摆法再拍，拍 ~20 张
     （每张都让所有工具、尤其绝缘电阻测试仪清晰露出来）
  4) 按【Q】或【ESC】退出
"""
import cv2
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "training" / "new_shots"
OUT.mkdir(parents=True, exist_ok=True)


def imwrite_unicode(path, img):
    cv2.imencode(".jpg", img)[1].tofile(str(path))


def main():
    # 从已有最大编号接着存，避免覆盖
    existing = sorted(OUT.glob("cam_*.jpg"))
    n = (int(existing[-1].stem.split("_")[1]) + 1) if existing else 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ 打不开摄像头：确认已关掉 run.py、摄像头没被别的程序占用")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("✅ 摄像头已开。摆好工具按【空格】拍照，按【Q】退出。")
    print(f"   保存到: {OUT}")

    shots = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # 预览上叠加提示（英文，cv2 不支持中文）
        view = frame.copy()
        cv2.putText(view, f"SPACE=capture  Q=quit   saved this run: {shots}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("capture (SPACE=shot, Q=quit)", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            path = OUT / f"cam_{n:04d}.jpg"
            imwrite_unicode(path, frame)
            print(f"  📸 {path.name}")
            n += 1
            shots += 1
        elif key in (ord('q'), ord('Q'), 27):  # q / ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    total = len(list(OUT.glob("cam_*.jpg")))
    print(f"\n✅ 本次拍了 {shots} 张，data/new_shots 共 {total} 张。")
    print("   拍够了就告诉我，我来预画框 + 合并重训。")


if __name__ == "__main__":
    main()
