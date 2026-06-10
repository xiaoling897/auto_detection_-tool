"""摄像头探测：枚举设备号 0~3，报告每个能否打开、分辨率、读帧是否有真画面。
换新摄像头打不开时用它定位——确认是"没认到/认错号/分辨率不对/出流慢"哪一种。
跑法：python scripts/probe_camera.py
"""
import time
import cv2

BACKENDS = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]


def probe(index, name, flag):
    cap = cv2.VideoCapture(index, flag)
    if not cap.isOpened():
        cap.release()
        return f"  [{name}] 设备号 {index}: 打不开"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cc = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip()

    # 读 10 帧测真实出流：耗时 + 画面方差（std<6 基本是黑屏/灰屏假画面）
    ok, std, t0, frames = False, 0.0, time.time(), 0
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            frames += 1
            std = float(frame.std())
            if std > 6.0:
                ok = True
        else:
            time.sleep(0.05)
    dt = (time.time() - t0) * 1000
    cap.release()

    verdict = "✅ 有真画面" if ok else ("⚠️ 黑/灰屏(std低)" if frames else "❌ 读不到帧")
    return (f"  [{name}] 设备号 {index}: 打开成功 默认 {w}x{h} 格式={cc or '?'} "
            f"读{frames}/10帧 {dt:.0f}ms std={std:.1f} → {verdict}")


def main():
    print("枚举摄像头（请确保摄像头已接 USB-C 进电脑、ON/OFF 拨 ON、必要时插 DC 12V）...\n")
    for name, flag in BACKENDS:
        print(f"=== 后端 {name} ===")
        for index in range(4):
            print(probe(index, name, flag))
        print()
    print("解读：")
    print("  · 全部'打不开' → Windows 没认到，是接线/供电/开关问题，跟程序无关")
    print("  · 某号'有真画面'但程序用不了 → 程序探测逻辑/分辨率需要按这个号和分辨率调")
    print("  · 4K 默认分辨率(如 3840x2160) + 读帧很慢 → 需要在打开后降到 1280x720 或换 MJPG")


if __name__ == "__main__":
    main()
