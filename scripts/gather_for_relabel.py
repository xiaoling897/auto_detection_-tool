"""把项目里所有"曾经标注过的图片"（AI 自动标注 + 手工标注）连同标签，
汇总到一个新文件夹 data/training/relabel/，供逐张重审/重标。

- 来源：data/training/{autolabel, label_multi, label_full, label_new, label_new2, label_new3}
- 去重：按文件名(stem)去重（autolabel 已含 new_*/b2_*，避免重复审同一张）
- 标签：labels/<stem>.txt 优先；没有就找 .xml(PascalVOC) 转成 YOLO .txt
        （.xml 里的类名支持中文或英文，自动映射到 class_map.json 的索引）
- 输出：relabel/images/、relabel/labels/、relabel/classes.txt（labelImg YOLO 模式用）

用法：  python scripts/gather_for_relabel.py
之后：  python scripts/open_labelimg.py relabel   # 在 relabel 文件夹里逐张标
"""
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "training"
OUT = TRAIN / "relabel"
SOURCES = ["autolabel", "label_multi", "label_full",
           "label_new", "label_new2", "label_new3"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def build_class_index():
    cm = json.load(open(ROOT / "data" / "yolo" / "class_map.json", encoding="utf-8"))
    eng = list(cm.keys())
    names = list(cm.values())                        # classes.txt 用中文显示名（顺序=索引）
    idx = {en: i for i, en in enumerate(eng)}        # 英文类名 -> 索引
    for i, (en, cn) in enumerate(cm.items()):
        idx[cn] = i                                  # 中文显示名 -> 同一索引
    return names, idx


def xml_to_yolo(xml_path, name_to_idx):
    """PascalVOC .xml -> YOLO 文本（class_id xc yc w h，归一化）。"""
    try:
        root = ET.parse(xml_path).getroot()
        w = float(root.find("size/width").text)
        h = float(root.find("size/height").text)
    except Exception:
        return ""
    if w <= 0 or h <= 0:
        return ""
    lines = []
    for obj in root.findall("object"):
        nm = (obj.findtext("name") or "").strip()
        cid = name_to_idx.get(nm)
        if cid is None:
            continue
        b = obj.find("bndbox")
        x1 = float(b.findtext("xmin")); y1 = float(b.findtext("ymin"))
        x2 = float(b.findtext("xmax")); y2 = float(b.findtext("ymax"))
        xc = (x1 + x2) / 2 / w; yc = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w; bh = (y2 - y1) / h
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return ("\n".join(lines) + "\n") if lines else ""


def find_label(stem, src, name_to_idx):
    """返回该图的 YOLO 标签文本（没有则空字符串）。"""
    txt = src / "labels" / f"{stem}.txt"
    if txt.exists():
        return txt.read_text(encoding="utf-8")
    for xml in (src / "labels" / f"{stem}.xml", src / "images" / f"{stem}.xml"):
        if xml.exists():
            return xml_to_yolo(xml, name_to_idx)
    return ""


def main():
    names, name_to_idx = build_class_index()
    out_img = OUT / "images"
    out_lbl = OUT / "labels"
    if OUT.exists():
        shutil.rmtree(OUT)
    out_img.mkdir(parents=True)
    out_lbl.mkdir(parents=True)

    seen = set()
    total, with_box, empty = 0, 0, 0
    per_src = {}
    for s in SOURCES:
        src = TRAIN / s
        imgdir = src / "images"
        if not imgdir.is_dir():
            continue
        cnt = 0
        for img in sorted(imgdir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            if img.stem in seen:
                continue
            seen.add(img.stem)
            lbl = find_label(img.stem, src, name_to_idx)
            shutil.copy(img, out_img / img.name)
            (out_lbl / f"{img.stem}.txt").write_text(lbl, encoding="utf-8")
            total += 1
            cnt += 1
            if lbl.strip():
                with_box += 1
            else:
                empty += 1
        per_src[s] = cnt

    # classes.txt（中文显示名，labelImg 选类别时显示中文；YOLO 模式保存目录也放一份）
    classes_txt = "\n".join(names) + "\n"
    (OUT / "classes.txt").write_text(classes_txt, encoding="utf-8")
    (out_lbl / "classes.txt").write_text(classes_txt, encoding="utf-8")

    print("=== 汇总完成 ===")
    for s, c in per_src.items():
        print(f"  {s}: 新增 {c} 张")
    print(f"  合计 {total} 张（{with_box} 张有框待审, {empty} 张空/待标）")
    print(f"  输出目录: {OUT}")
    print("  下一步: python scripts/open_labelimg.py relabel")


if __name__ == "__main__":
    main()
