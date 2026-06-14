"""把 labelImg 的 CreateML .json 标注转成 YOLO .txt。

CreateML 格式：coordinates 的 x/y 是像素中心点，width/height 是像素宽高。
YOLO 格式：class_id cx cy w h（全部除以图像宽高，归一化到 0~1）。

用法：python scripts/createml_to_yolo.py data/training/new_shots3
转换后 .json 移到同目录 _createml_json_backup/ 备份（不删）。
"""
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "data/training/new_shots3").resolve()
CLASSES = (DIR / "classes.txt").read_text(encoding="utf-8").split()
NAME2ID = {n: i for i, n in enumerate(CLASSES)}

backup = DIR / "_createml_json_backup"
backup.mkdir(exist_ok=True)

jsons = sorted(DIR.glob("*.json"))
print(f"📋 classes.txt: {CLASSES}")
print(f"🔍 找到 {len(jsons)} 个 json\n")

total_boxes = 0
converted = 0
unknown_labels = set()
problems = []

for jf in jsons:
    data = json.loads(jf.read_text(encoding="utf-8"))
    # CreateML 是一个 list，每个元素 {"image":..., "annotations":[...]}
    entry = data[0] if isinstance(data, list) else data
    img_name = entry.get("image", jf.stem + ".jpg")
    img_path = DIR / img_name
    if not img_path.exists():
        img_path = jf.with_suffix(".jpg")
    if not img_path.exists():
        problems.append(f"{jf.name}: 找不到对应图片")
        continue

    with Image.open(img_path) as im:
        W, H = im.size

    lines = []
    for ann in entry.get("annotations", []):
        label = ann["label"]
        if label not in NAME2ID:
            unknown_labels.add(label)
            problems.append(f"{jf.name}: 未知标签 '{label}'，跳过")
            continue
        cid = NAME2ID[label]
        c = ann["coordinates"]
        cx, cy, w, h = c["x"] / W, c["y"] / H, c["width"] / W, c["height"] / H
        # 夹到 0~1
        cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
        w, h = min(max(w, 0), 1), min(max(h, 0), 1)
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        total_boxes += 1

    out = jf.with_suffix(".txt")
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    shutil.move(str(jf), str(backup / jf.name))
    converted += 1
    print(f"  ✅ {jf.stem}: {len(lines)} 框 (图 {W}x{H})")

print(f"\n转换完成：{converted} 个文件，{total_boxes} 个框")
if unknown_labels:
    print(f"⚠️  未知标签（不在 classes.txt）：{unknown_labels}")
if problems:
    print("⚠️  问题：")
    for p in problems:
        print("   " + p)
print(f"📦 原 json 已备份到：{backup}")
