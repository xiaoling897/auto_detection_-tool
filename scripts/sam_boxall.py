"""给多工具图用 SAM 预画出"所有物体"的框（不分类）——人工只需点选每个框的类别。

为什么单独做多工具图：单工具图靠文件夹名能自动定类，多工具图不能（一张图里多个不同类），
所以这步必须人工指认每个框是什么工具。但"画框"这件体力活由 SAM 代劳，人只点类别。

输入：  C:\\Users\\acad1\\Desktop\\tool_dataset\\全部在一起照片\\*.jpg
输出：  data/label_multi/
          images/*.jpg     原图
          labels/*.txt     YOLO 框，class 一律先写 0（占位）——你在 labelImg 里改成真类别
          preview/*.jpg     画了编号框的预览，先看一眼 SAM 画得全不全
          classes.txt       类别清单（labelImg 用）

用法：python scripts/sam_boxall.py
然后用 labelImg 打开 data/label_multi/images，逐张把每个框的类别点对、删掉画歪的。
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# 默认处理"全部在一起照片"；也可命令行指定：python sam_boxall.py <源文件夹> <输出文件夹>
SOURCE = Path(r"C:\Users\acad1\Desktop\tool_dataset") / "全部在一起照片"
OUT = PROJECT_ROOT / "data" / "training" / "label_multi"
if len(sys.argv) > 1:
    SOURCE = Path(sys.argv[1])
if len(sys.argv) > 2:
    OUT = Path(sys.argv[2])
# 输出文件名前缀（按输出目录名区分，避免不同批次重名冲突）
PREFIX = OUT.name.replace("label_", "") or "img"
CLASS_MAP_PATH = PROJECT_ROOT / "data" / "yolo" / "class_map.json"

# 物体框过滤（多工具：每个工具比单工具图里小，范围放宽下限）
MIN_AREA = 0.008    # 下限：≥0.8% 才算一个物体（滤碎屑）
MAX_AREA = 0.45     # 上限：单个工具不会超过画面 45%
BG_SPAN, BG_OTHER = 0.93, 0.6   # 背景块过滤（跨满一维+另一维也大）
MIN_TEXTURE = 6.0   # 平均梯度下限：低于此当光滑背景丢
IOU_DEDUP = 0.6     # 去重：与已保留框 IoU 超此值则丢
MAX_BOXES = 12      # 每图最多保留框数（按得分）


def imread(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def imwrite(p, img): cv2.imencode(Path(p).suffix or ".jpg", img)[1].tofile(str(p))


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def collect_boxes(masks, grad, W, H):
    """从 SAM masks 里收集所有"像物体"的框，去重后按得分排序返回 [(x1,y1,x2,y2), ...]"""
    cands = []
    for mk in masks:
        sel = mk > 0.5
        ys, xs = np.where(sel)
        if len(xs) == 0:
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        frac = (x2 - x1) * (y2 - y1) / (W * H)
        if frac < MIN_AREA or frac > MAX_AREA:
            continue
        wf, hf = (x2 - x1) / W, (y2 - y1) / H
        if (wf >= BG_SPAN and hf >= BG_OTHER) or (hf >= BG_SPAN and wf >= BG_OTHER):
            continue
        texture = float(grad[sel].mean())
        if texture < MIN_TEXTURE:
            continue
        score = (frac ** 0.3) * texture     # 物体越大越有纹理越靠前
        cands.append((score, (x1, y1, x2, y2)))
    cands.sort(reverse=True)
    kept = []
    for _, box in cands:
        if all(iou(box, k) < IOU_DEDUP for k in kept):
            kept.append(box)
        if len(kept) >= MAX_BOXES:
            break
    return kept


def main():
    classes = list(json.load(open(CLASS_MAP_PATH, encoding="utf-8")).keys())
    if not SOURCE.exists():
        raise FileNotFoundError(f"找不到多工具文件夹: {SOURCE}")

    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ("images", "labels", "preview"):
        (OUT / s).mkdir(parents=True, exist_ok=True)
    (OUT / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    print("📥 加载 FastSAM-s.pt ...")
    from ultralytics import FastSAM
    model = FastSAM(str(PROJECT_ROOT / "models" / "FastSAM-s.pt"))

    imgs = sorted([p for p in SOURCE.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    print(f"🔍 {len(imgs)} 张多工具图")
    total_boxes = 0
    for i, p in enumerate(imgs):
        if i % 10 == 0 and i:
            print(f"  {i}/{len(imgs)}")
        img = imread(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3),
                             cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3))
        r = model(img, retina_masks=True, conf=0.4, iou=0.9, verbose=False)[0]
        masks = r.masks.data.cpu().numpy() if r.masks is not None else []
        boxes = collect_boxes(masks, grad, W, H)

        stem = f"{PREFIX}_{i:04d}"
        imwrite(OUT / "images" / f"{stem}.jpg", img)
        lines, preview = [], img.copy()
        for j, (x1, y1, x2, y2) in enumerate(boxes):
            xc = (x1 + x2) / 2 / W; yc = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W; bh = (y2 - y1) / H
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")  # 占位类别 0
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(preview, str(j + 1), (x1 + 5, y1 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        (OUT / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        imwrite(OUT / "preview" / f"{stem}.jpg", preview)
        total_boxes += len(boxes)

    print(f"\n✅ {len(imgs)} 张图，预画 {total_boxes} 个框（平均 {total_boxes/max(len(imgs),1):.1f} 个/图）")
    print(f"   所有框类别都是占位 0，需要你在 labelImg 里逐个改成真类别。")
    print(f"   下一步: labelImg {OUT/'images'} {OUT/'classes.txt'}")


if __name__ == "__main__":
    main()
