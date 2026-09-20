"""一键启动 labelImg 标注（中文界面），自动建批次目录、导入照片、指向 classes.txt。

【最常用】有一组新照片要标注，一条命令搞定（建目录 + 导图 + 启动）：
    python scripts/open_labelimg.py --from "C:\\Users\\acad1\\Desktop\\new_photo"
  自动新建下一个批次号（relabel9 / relabel10 ...），把照片拷进去，然后打开 labelImg。

【指定批次名】
    python scripts/open_labelimg.py relabel9 --from "D:\\照片"   # 建 relabel9 并导图
    python scripts/open_labelimg.py relabel9                     # 接着标 relabel9（断点继续）

【为什么批次目录必须叫 relabel<数字>】
  打包训练集的 scripts/prepare_relabel.py 只扫 data/training/relabel* 开头的目录。
  叫别的名字（label_new4 之类）标了也白标，不会进训练集。本脚本会帮你挡住这个坑。

第一次用先装标注软件：pip install labelImg
标完接着跑：
    python scripts/prepare_relabel.py     # 合并所有批次 → yolo_dataset
    python scripts/train_until_done.py    # 训练（跑之前先归档旧的 runs/tool_yolo）
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "training"
CLASS_MAP = ROOT / "data" / "yolo" / "class_map.json"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# labelImg 的界面语言靠 libs/stringBundle.py 里的 locale.getlocale() 决定：
# 中文 Windows 上它返回 'Chinese (Simplified)_China'，按 [^a-zA-Z] 切出来是
# ':/strings-Chinese'，而内置资源里只有 ':/strings-zh-CN'——匹配不上就退回英文。
# 这里在进程内把 getlocale 顶成 'zh_CN'，切出 ':/strings-zh' + ':/strings-zh-CN'，
# 后者命中，界面就是中文。必须先 import libs.resources，否则 qrc 没注册、字符串全空。
_ZH_LAUNCH_CODE = (
    "import locale;"
    "locale.getlocale=lambda *a, **k: ('zh_CN', 'UTF-8');"
    "from libs.resources import *;"
    "from labelImg.labelImg import main;"
    "main()"
)


def next_batch_name():
    """扫描已有的 relabel<数字> 批次，返回下一个没被占用的名字。"""
    used = set()
    for p in TRAIN_DIR.glob("relabel*"):
        m = re.fullmatch(r"relabel(\d*)", p.name)
        if p.is_dir() and m:
            used.add(int(m.group(1)) if m.group(1) else 1)
    n = 2
    while n in used:
        n += 1
    return f"relabel{n}"


def write_classes(path):
    """classes.txt 从 class_map.json 生成——那是工具清单的唯一权威来源，
    顺序即类别编号(0~8)，绝不能和训练时对不上。"""
    names = list(json.load(open(CLASS_MAP, encoding="utf-8")).values())
    path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return names


def existing_stems():
    """所有已存在批次里的图片文件名（不含扩展名）。

    prepare_relabel.py 按文件名去重、先扫到的批次赢，所以重名的新图标了也会被忽略。
    导入时直接跳过，省得白标。"""
    stems = set()
    for src in TRAIN_DIR.glob("relabel*"):
        d = src / "images"
        if d.is_dir():
            stems |= {p.stem for p in d.iterdir() if p.suffix.lower() in IMG_EXT}
    return stems


def import_photos(src_dir, img_dir):
    """把照片拷进批次目录。返回 (导入数, 跳过的重名数)。

    ⚠️ 必须用 copyfile 而不是 copy2：prepare_relabel.py 判断"这张人工标过"的依据是
    标签 .txt 比图片新 60 秒以上，保留原始 mtime 会让新标的标签判不达标、被漏掉。"""
    src = Path(src_dir).expanduser()
    if not src.is_dir():
        print(f"❌ 找不到照片目录：{src}")
        return None
    photos = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    if not photos:
        print(f"❌ {src} 里没有图片（支持 {'/'.join(IMG_EXT)}）")
        return None
    taken = existing_stems()
    added = dup = 0
    for p in photos:
        if p.stem in taken:
            dup += 1
            continue
        shutil.copyfile(p, img_dir / p.name)
        added += 1
    return added, dup


def ensure_yolo_format():
    """把 labelImg 默认保存格式预设成 YOLO（写 ~/.labelImgSettings.pkl）。

    labelImg 把"上次用的格式"存这个文件里，启动时读它。预设成 YOLO 后，
    打开就是 YOLO，不用每次手动点左下角切格式（默认是 PascalVOC，会存成 .xml）。"""
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
    except Exception as e:
        print(f"(写 labelImg 设置失败，需手动切 YOLO：{e})")


def find_labelimg():
    """返回启动 labelImg 的命令前缀；找不到返回 None。

    优先用当前解释器 + 中文补丁启动（见 _ZH_LAUNCH_CODE）；
    装在别的环境里时退回 labelImg.exe（界面会是英文）。"""
    try:
        import labelImg  # noqa: F401  只探测能否 import
        return [sys.executable, "-c", _ZH_LAUNCH_CODE]
    except Exception:
        pass
    exe = shutil.which("labelImg") or str(Path(sys.executable).parent / "Scripts" / "labelImg.exe")
    if Path(exe).is_file() or shutil.which("labelImg"):
        print("(当前解释器里没有 labelImg 包，退回 exe 启动，界面可能是英文)")
        return [exe]
    return None


def main():
    ap = argparse.ArgumentParser(description="启动 labelImg 标注（中文界面）")
    ap.add_argument("batch", nargs="?", help="批次目录名，如 relabel9；省略则自动取下一个")
    ap.add_argument("--from", dest="src", help="照片来源目录，会拷贝进批次目录")
    args = ap.parse_args()

    batch = args.batch
    if batch is None:
        if not args.src:
            ap.error("要么给批次名（python scripts/open_labelimg.py relabel9），"
                     "要么给照片目录（--from <目录>）")
        batch = next_batch_name()
        print(f"📁 自动分配批次：{batch}")
    if not re.fullmatch(r"relabel\d*", batch):
        print(f"⚠️  批次名 '{batch}' 不是 relabel<数字> 格式。")
        print("    prepare_relabel.py 只收 relabel* 开头的目录，别的名字标了不会进训练集。")
        return

    base = TRAIN_DIR / batch
    existed = base.is_dir()          # 本来就有？没有的话一张图都没导入时要清理掉
    img_dir, lbl_dir = base / "images", base / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    classes = base / "classes.txt"
    names = write_classes(classes)
    # labelImg 的 YOLO 模式会在"保存目录"里读写 classes.txt，没有就崩
    shutil.copyfile(classes, lbl_dir / "classes.txt")

    if args.src:
        r = import_photos(args.src, img_dir)
        if r is None:
            return
        added, dup = r
        print(f"📷 导入 {added} 张到 {img_dir}" + (f"（跳过 {dup} 张与已有批次重名的）" if dup else ""))

    total = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if total == 0:
        if not existed:
            # 这个批次是本次新建的、又一张图都没进来（照片全重名），别留个空壳占着批次号
            shutil.rmtree(base, ignore_errors=True)
            print(f"⚠️  一张都没导入，已撤销刚建的 {batch}。")
            print("    原因：照片和已有批次重名，标了也会被去重忽略。")
            print("    请改用一批新拍的照片，或把重复的挑掉再试。")
        else:
            print(f"⚠️  {img_dir} 里没有照片。")
            print("    把照片拷进去，或者重跑并加上 --from <照片目录>")
        return

    done = sum(1 for p in lbl_dir.glob("*.txt")
               if p.name != "classes.txt" and p.stat().st_size > 0)
    ensure_yolo_format()

    cmd = find_labelimg()
    if cmd is None:
        print("❌ 没找到 labelImg，请先安装：pip install labelImg")
        return

    cmd += [str(img_dir), str(classes), str(lbl_dir)]
    print(f"🏷  启动 labelImg（中文界面）：{batch}  共 {total} 张，已标 {done} 张")
    print(f"    类别({len(names)}): {' '.join(names)}")
    print("    标完一张记得 Ctrl+S；不想要的图直接跳过不存即可。")
    try:
        subprocess.Popen(cmd, cwd=str(ROOT))
    except Exception as e:
        print(f"❌ 启动失败：{e}")


if __name__ == "__main__":
    main()
