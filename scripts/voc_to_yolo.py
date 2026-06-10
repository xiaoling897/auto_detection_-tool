"""把 labelImg 存的 PascalVOC .xml 标注转成 YOLO .txt（写进 data/label_multi/labels/）。

labelImg 默认用 PascalVOC(.xml) 保存。本脚本把 label_multi 下所有 .xml（images/ 或 labels/ 里）
按 <name>(中文类名) → classes.txt 行号(0基) 转成 YOLO 格式，覆盖同名 .txt。

用法：python scripts/voc_to_yolo.py
"""
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent / "data" / "training" / "label_multi"
LBL = ROOT / "labels"


def main():
    classes = (ROOT / "classes.txt").read_text(encoding="utf-8").split()
    name_to_id = {n: i for i, n in enumerate(classes)}

    # 收集所有 xml（labels/ 优先；images/ 里的同名作补充）
    xmls = {}
    for sub in ("labels", "images"):
        for x in (ROOT / sub).glob("*.xml"):
            xmls.setdefault(x.stem, x)   # labels/ 先入，优先

    if not xmls:
        print("没找到任何 .xml，无需转换")
        return

    converted, unknown = 0, set()
    for stem, xml in sorted(xmls.items()):
        tree = ET.parse(xml)
        r = tree.getroot()
        W = int(r.findtext("size/width"))
        H = int(r.findtext("size/height"))
        lines = []
        for obj in r.findall("object"):
            name = obj.findtext("name")
            if name not in name_to_id:
                unknown.add(name)
                continue
            cid = name_to_id[name]
            b = obj.find("bndbox")
            x1 = float(b.findtext("xmin")); y1 = float(b.findtext("ymin"))
            x2 = float(b.findtext("xmax")); y2 = float(b.findtext("ymax"))
            xc = (x1 + x2) / 2 / W; yc = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W; bh = (y2 - y1) / H
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (LBL / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        converted += 1
        print(f"  {stem}: {len(lines)} 个框")

    print(f"\n✅ 转换 {converted} 张 → YOLO txt（写入 {LBL}）")
    if unknown:
        print(f"⚠️  这些类名不在 classes.txt、已跳过: {unknown}")


if __name__ == "__main__":
    main()
