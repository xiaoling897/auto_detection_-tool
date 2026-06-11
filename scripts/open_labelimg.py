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
# 可选参数：要标注的子文件夹名（data/training/ 下），默认 autolabel。
#   python scripts/open_labelimg.py            -> data/training/autolabel
#   python scripts/open_labelimg.py relabel    -> data/training/relabel
_SUB = sys.argv[1] if len(sys.argv) > 1 else "autolabel"
BASE = ROOT / "data" / "training" / _SUB
IMG_DIR = BASE / "images"
LBL_DIR = BASE / "labels"
CLASSES = BASE / "classes.txt"


def ensure_yolo_format():
    """把 labelImg 默认保存格式预设成 YOLO（写 ~/.labelImgSettings.pkl）。

    labelImg 把"上次用的格式"存这个文件里，启动时读它。预设成 YOLO 后，
    打开就是 YOLO，不用每次手动点左下角切格式（默认是 PascalVOC，会存成 .xml）。
    """
    try:
        import pickle
        from libs.labelFile import LabelFileFormat
    except Exception as e:
        print(f"(无法预设 YOLO 默认，需手动切左下角格式：{e})")
        return
    pkl = Path.home() / ".labelImgSettings.pkl"
    data = {}
    if pkl.is_file():
        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f)      # 尽量保留其它设置（窗口位置等）
        except Exception:
            data = {}
    data["labelFileFormat"] = LabelFileFormat.YOLO
    try:
        with open(pkl, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print("✅ 已把 labelImg 默认格式设为 YOLO（打开即是，不用手动切）")
    except Exception as e:
        print(f"(写 labelImg 设置失败，需手动切 YOLO：{e})")


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
    # labelImg 的 YOLO 模式会在“保存目录”里读写 classes.txt，没有就崩。
    # 这里把项目的 classes.txt 同步一份到 labels/，避免启动报 FileNotFoundError。
    if CLASSES.is_file():
        shutil.copy(CLASSES, LBL_DIR / "classes.txt")
    if not IMG_DIR.is_dir() or not any(IMG_DIR.glob("*.jpg")):
        print(f"⚠️  图片目录为空或不存在：{IMG_DIR}")
        print("    先把要标注的照片(.jpg)放进去再运行本脚本。")
        return

    ensure_yolo_format()   # 预设默认格式 YOLO

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
