"""把 relabel/ 里"已经过目处理"的图（已标注 + 跳过的）整体归档到 relabel_done/，
这样重开 labelImg 时 relabel/ 只剩没碰过的，直接从断点继续，不用每次从头翻。

判断"已标注"：标签 .txt 的修改时间明显晚于图片（汇总时图和标签同时生成，时间几乎相同；
你手动 Ctrl+S 存的标签时间会晚很多）。比硬编码日期稳，可重复运行。

归档边界：按文件名顺序，最后一张"已标注"的位置。它**之前**的全算"处理过"（标的+跳过的），
一起搬到 relabel_done/；它**之后**的留在 relabel/ 继续标。

训练时只用 relabel_done/ 里"真正标注过"的（跳过的旧标签会被排除，见 prepare 阶段）。

用法：  python scripts/archive_labeled.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 可选参数：要归档的批次文件夹名（默认 relabel）。归档到 <名>_done。
#   python scripts/archive_labeled.py            -> relabel  -> relabel_done
#   python scripts/archive_labeled.py relabel3   -> relabel3 -> relabel3_done
_SUB = sys.argv[1] if len(sys.argv) > 1 else "relabel"
BASE = ROOT / "data" / "training" / _SUB
DONE = ROOT / "data" / "training" / f"{_SUB}_done"
IMG, LBL = BASE / "images", BASE / "labels"
MARGIN = 60.0   # 标签比图片晚 >60s 视为"人工存过"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def is_labeled(stem):
    img = next((IMG / f"{stem}{e}" for e in IMG_EXT if (IMG / f"{stem}{e}").exists()), None)
    txt = LBL / f"{stem}.txt"
    if img is None or not txt.exists():
        return False
    return txt.stat().st_mtime > img.stat().st_mtime + MARGIN


def main():
    stems = sorted(p.stem for p in IMG.iterdir() if p.suffix.lower() in IMG_EXT)
    if not stems:
        print("relabel/images 为空，没什么可归档的。")
        return
    touched = [i for i, s in enumerate(stems) if is_labeled(s)]
    if not touched:
        print("还没有已标注的图（没有可归档的）。")
        return

    boundary = max(touched)
    prefix = stems[:boundary + 1]
    n_labeled = len(touched)
    n_skip = len(prefix) - n_labeled

    (DONE / "images").mkdir(parents=True, exist_ok=True)
    (DONE / "labels").mkdir(parents=True, exist_ok=True)
    for s in prefix:
        for e in IMG_EXT:
            f = IMG / f"{s}{e}"
            if f.exists():
                shutil.move(str(f), str(DONE / "images" / f.name))
                break
        t = LBL / f"{s}.txt"
        if t.exists():
            shutil.move(str(t), str(DONE / "labels" / t.name))

    cls = BASE / "classes.txt"
    if cls.exists():
        shutil.copy(cls, DONE / "classes.txt")
        shutil.copy(cls, DONE / "labels" / "classes.txt")

    remaining = sum(1 for p in IMG.iterdir() if p.suffix.lower() in IMG_EXT)
    print(f"已归档处理过的 {len(prefix)} 张 -> relabel_done/"
          f"（真标注 {n_labeled} 张, 跳过 {n_skip} 张）")
    print(f"relabel_done/ 累计图片: "
          f"{sum(1 for p in (DONE/'images').iterdir() if p.suffix.lower() in IMG_EXT)} 张")
    print(f"relabel/ 还剩 {remaining} 张待标 —— 重开 labelImg 直接从断点继续")


if __name__ == "__main__":
    main()
