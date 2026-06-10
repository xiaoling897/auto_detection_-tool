"""一键启动 labelImg 标注，自动指向本项目的标注目录 + classes.txt。

用法：
    python scripts/open_labelimg.py

注意：labelImg 是独立的标注软件，不随本项目分发。第一次用先装：
    pip install labelImg
照片要先放到 data/training/autolabel/images/ 下（标签会存到同级 labels/）。
labelImg 打开后记得：左下角格式切到 YOLO；Change Save Dir 选 labels 目录。
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "training" / "autolabel" / "images"
LBL_DIR = ROOT / "data" / "training" / "autolabel" / "labels"
CLASSES = ROOT / "data" / "training" / "autolabel" / "classes.txt"


def find_labelimg():
    """返回启动 labelImg 的命令前缀；找不到返回 None。"""
    exe = shutil.which("labelImg")
    if exe:
        return [exe]
    cand = Path(sys.executable).parent / "Scripts" / "labelImg.exe"
    if cand.is_file():
        return [str(cand)]
    return None


def main():
    LBL_DIR.mkdir(parents=True, exist_ok=True)
    if not IMG_DIR.is_dir() or not any(IMG_DIR.glob("*.jpg")):
        print(f"⚠️  图片目录为空或不存在：{IMG_DIR}")
        print("    先把要标注的照片(.jpg)放进去再运行本脚本。")
        return

    cmd = find_labelimg()
    if cmd is None:
        print("❌ 没找到 labelImg。请先安装：")
        print("       pip install labelImg")
        return

    # labelImg 位置参数：图片目录、预定义类别文件、默认保存目录
    cmd += [str(IMG_DIR), str(CLASSES), str(LBL_DIR)]
    print(f"🏷  启动 labelImg：{IMG_DIR}")
    print("    提醒：左下角格式切到 YOLO；类别从 classes.txt 读取；标签存到 labels/。")
    try:
        subprocess.Popen(cmd)
    except Exception as e:
        print(f"❌ 启动失败：{e}\n   手动启动：labelImg {IMG_DIR} {CLASSES}")


if __name__ == "__main__":
    main()
