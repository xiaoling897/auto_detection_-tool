"""用摄像头在实际监控位置批量采集训练素材。

和 capture_templates.py 的区别：
  - capture_templates 是 ORB 模板采集（旧 SIFT 方案），一个工具拍一张
  - 本脚本是给 YOLO 攒训练集：同一工具连拍几十上百张，边拍边换角度/光线/位置

输出直接落到 auto_label_with_yoloworld.py 读取的目录：
  C:/Users/<你>/Desktop/tool_dataset/<工具中文名>/cam_xxx.jpg
采完无缝接：
  python scripts/auto_label_with_yoloworld.py   # 自动标注
  python scripts/train_yolo.py                  # 训练

操作（窗口聚焦时按键）：
  数字 1-9   切换当前要拍的工具
  空格        手动拍一张
  c          开/关「自动连拍」（开着就持续采样，把工具在镜头前转着摆即可）
  q 或 ESC   退出（自动保存）

关键用法建议：在摄像头【实际安装位置】拍，让训练图 = 将来检测时看到的画面，
这是解决「摄像头漏检」最有效的一步。开着自动连拍后，慢慢转动/移动工具、
改变光线和远近、偶尔用别的东西挡住一角，让样本尽量多样。
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============ 配置 ============
# 输出根目录：和 auto_label_with_yoloworld.py 的 SOURCE_DIR 保持一致
OUTPUT_ROOT = Path.home() / "Desktop" / "tool_dataset"

# 工具列表：文件夹名必须和 auto_label_with_yoloworld.py 的 FOLDER_TO_CLASS 的 key 一字不差
TOOLS = [
    "绝缘钢丝钳",
    "数字万用表",   # 注意：auto_label 里这个文件夹叫「万用表」，见下方 FOLDER_NAME_FIX
    "绝缘测试仪",
    "网线测线仪",
    "激兆测距仪",   # auto_label 里就是这个名（带 typo），保持一致
    "电工胶带",
    "磁力线坠",
    "胎压测试仪",
    "卷尺",
]

# 个别显示名 → 实际文件夹名（对齐 auto_label_with_yoloworld.py 已有的文件夹）
FOLDER_NAME_FIX = {
    "数字万用表": "万用表",
}

CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720      # 和 gui.py 摄像头分辨率一致
AUTO_CAPTURE_INTERVAL = 0.25      # 自动连拍间隔（秒）→ 约 4 张/秒，避免存一堆重复帧
JPEG_QUALITY = 92

_CN_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]


def load_font(size):
    for p in _CN_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def imwrite_unicode(path: Path, img) -> bool:
    """中文路径安全写图（cv2.imwrite 在中文路径下会静默失败）"""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def folder_for(tool: str) -> Path:
    name = FOLDER_NAME_FIX.get(tool, tool)
    d = OUTPUT_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def existing_count(tool: str) -> int:
    d = folder_for(tool)
    return sum(1 for p in d.iterdir()
               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def draw_overlay(frame, tool, session_saved, total_in_folder, auto_on, font, font_small):
    """用 PIL 画中文叠层（cv2.putText 不支持中文）"""
    h, w = frame.shape[:2]
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)

    # 顶部半透明信息条
    d.rectangle([(0, 0), (w, 96)], fill=(0, 0, 0))
    rec = "● 自动连拍中" if auto_on else "○ 手动模式（按 c 开自动连拍）"
    rec_color = (255, 80, 80) if auto_on else (180, 180, 180)
    d.text((16, 10), f"当前工具：{tool}", font=font, fill=(0, 255, 136))
    d.text((16, 44), rec, font=font_small, fill=rec_color)
    d.text((520, 10), f"本次已拍：{session_saved}", font=font_small, fill=(255, 255, 255))
    d.text((520, 40), f"该工具总数：{total_in_folder}", font=font_small, fill=(255, 204, 0))

    # 底部按键提示
    d.rectangle([(0, h - 34), (w, h)], fill=(0, 0, 0))
    d.text((16, h - 30),
           "1-9 切换工具   空格 拍一张   c 自动连拍   q/ESC 退出",
           font=font_small, fill=(200, 200, 200))

    # 居中取景参考框（帮你把工具摆在画面中间，自动标注的兜底框更准）
    cx1, cy1, cx2, cy2 = int(w * 0.18), int(h * 0.18), int(w * 0.82), int(h * 0.86)
    d.rectangle([(cx1, cy1), (cx2, cy2)], outline=(0, 200, 255), width=1)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else CAMERA_INDEX

    print("=" * 56)
    print(" 摄像头批量采集训练素材")
    print(f" 输出目录：{OUTPUT_ROOT}")
    print(f" 可采集工具：{', '.join(TOOLS)}")
    print("=" * 56)
    print(" 正在打开摄像头（Windows 下可能要等 5-9 秒）...")

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print(f"✗ 无法打开摄像头（index={cam_index}）。换个 index 试试：python scripts/capture_camera_dataset.py 1")
        return

    font = load_font(26)
    font_small = load_font(18)

    tool_idx = 0
    auto_on = False
    last_auto = 0.0
    session_saved = {t: 0 for t in TOOLS}
    session_ts = time.strftime("%Y%m%d_%H%M%S")
    seq = 0  # 全局递增序号，保证文件名不撞

    win = "Capture Dataset"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, FRAME_W, FRAME_H)

    print("✓ 摄像头已就绪，开始采集。窗口聚焦时按键操作。\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ 读取帧失败")
                break

            tool = TOOLS[tool_idx]
            now = time.time()

            do_save = False
            # 自动连拍：到间隔就存一张
            if auto_on and (now - last_auto) >= AUTO_CAPTURE_INTERVAL:
                do_save = True
                last_auto = now

            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):  # q / ESC
                break
            elif key == ord(' '):      # 手动拍一张
                do_save = True
            elif key == ord('c'):      # 切换自动连拍
                auto_on = not auto_on
                last_auto = now
                print(f"  自动连拍：{'开' if auto_on else '关'}")
            elif ord('1') <= key <= ord('9'):
                ni = key - ord('1')
                if ni < len(TOOLS):
                    tool_idx = ni
                    print(f"  切到工具：{TOOLS[tool_idx]}（已有 {existing_count(TOOLS[tool_idx])} 张）")

            if do_save:
                seq += 1
                fname = f"cam_{session_ts}_{seq:05d}.jpg"
                if imwrite_unicode(folder_for(tool) / fname, frame):
                    session_saved[tool] += 1

            disp = draw_overlay(
                frame, tool, session_saved[tool], existing_count(tool),
                auto_on, font, font_small,
            )
            cv2.imshow(win, disp)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 56)
    print(" 采集结束，本次新增：")
    total = 0
    for t in TOOLS:
        if session_saved[t]:
            print(f"   {t}: +{session_saved[t]} 张（累计 {existing_count(t)}）")
            total += session_saved[t]
    print(f" 合计新增 {total} 张 → {OUTPUT_ROOT}")
    print("=" * 56)
    print(" 下一步：")
    print("   python scripts/auto_label_with_yoloworld.py   # 自动标注")
    print("   python scripts/train_yolo.py                  # 训练")
    print("=" * 56)


if __name__ == "__main__":
    main()
