"""用当前模型(best.pt)给某个 relabel 批次自动"预标"，生成 YOLO 草稿框，供人工复核。

策略（按文件名前缀 "<工具中文名>__原名.jpg" 区分）：
  - 单工具图（前缀是某个工具名）：用模型找出最显眼的框作"位置"，
    **类别直接强制成文件夹对应的工具**（即使模型认错也不怕，因为文件夹已告诉我们是哪件）。
    模型一个框都没出 → 留空，复核时手画。
  - 多工具图（前缀 "全部在一起" 或无前缀）：用模型的检测结果（类别+框）原样写入，复核时改错的。

之后用 labelImg 打开复核：单工具图基本只需看框准不准；多工具图改几个认错的。

用法：  python scripts/autolabel_with_model.py            # 默认 relabel3
        python scripts/autolabel_with_model.py relabel3
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils import imread_unicode  # noqa: E402

SUB = sys.argv[1] if len(sys.argv) > 1 else "relabel3"
BASE = ROOT / "data" / "training" / SUB
IMG, LBL = BASE / "images", BASE / "labels"
MODEL = ROOT / "data" / "yolo" / "best.pt"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
SINGLE_CONF = 0.15    # 单工具图：放低门槛，尽量给个框（类别反正强制成文件夹的）
MULTI_CONF = 0.35     # 多工具图：高一点，少些要删的误框
MULTI_PREFIX = "全部在一起"


def main():
    cm = json.load(open(ROOT / "data" / "yolo" / "class_map.json", encoding="utf-8"))
    cn_to_idx = {cn: i for i, cn in enumerate(cm.values())}   # 中文工具名 -> 类别索引

    from ultralytics import YOLO
    model = YOLO(str(MODEL))
    LBL.mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in IMG.iterdir() if p.suffix.lower() in IMG_EXT)
    n_single_box = n_single_empty = n_multi = 0
    for img in imgs:
        prefix = img.name.split("__")[0] if "__" in img.name else ""
        expected = cn_to_idx.get(prefix)            # None=多工具/未知前缀
        is_single = expected is not None and prefix != MULTI_PREFIX
        conf = SINGLE_CONF if is_single else MULTI_CONF

        arr = imread_unicode(str(img))              # 中文路径要用这个读
        lines = []
        if arr is not None:
            r = model(arr, conf=conf, iou=0.45, verbose=False)[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xywhn = r.boxes.xywhn.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                if is_single:
                    bi = int(confs.argmax())        # 最显眼的框 = 那件工具
                    x, y, w, h = xywhn[bi]
                    lines.append(f"{expected} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    n_single_box += 1
                else:
                    for (x, y, w, h), c in zip(xywhn, clss):
                        lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    n_multi += 1
            elif is_single:
                n_single_empty += 1
        (LBL / f"{img.stem}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

    print(f"预标完成：{SUB} 共 {len(imgs)} 张")
    print(f"  单工具-出框: {n_single_box} 张（类别已按文件夹强制改对，复核看框准不准）")
    print(f"  单工具-模型没出框(留空待手画): {n_single_empty} 张")
    print(f"  多工具-按模型检测预标: {n_multi} 张（复核改认错的）")
    print(f"  下一步: python scripts/open_labelimg.py {SUB}  复核")


if __name__ == "__main__":
    main()
