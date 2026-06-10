"""用 SAM(类别无关分割)给单工具文件夹自动打框 —— AI 辅助标注的"打草稿"环节。

思路（为什么这么干，见项目讨论）：
  - 单工具文件夹的"类别"由文件夹名给定，不需要 AI 去"认"工具（YOLO-World 认不出，已实测）。
  - AI 只需干"类别无关"的事：找出画面里那个最显眼、居中的物体在哪 → SAM 干这个最稳。
  - 但全自动会犯错（框太大/框错对象/多工具图分不清），所以本脚本带合理性过滤：
      只自动接受"单个、居中、大小正常(2%~80%)"的框；可疑的写进 review 清单让人重点看。

输入：  C:\\Users\\acad1\\Desktop\\tool_dataset\\<工具中文名>/*.jpg
输出：  data/autolabel/
          images/*.jpg          统一转 jpg，文件名带类别前缀
          labels/*.txt          YOLO 格式: class_id xc yc w h（归一化），可被 labelImg/Roboflow 读
          preview/*.jpg         画了框 + 标记(AUTO/REVIEW)的预览图，供快速人工抽查
          classes.txt           labelImg 需要的类别清单（英文，顺序=class_map.json 的 key 顺序）
          review_list.txt       被判为"需人工核对"的图片清单

用法：
    python scripts/sam_autolabel.py            # 跑全部 712 张（CPU 上较慢）
    python scripts/sam_autolabel.py 6          # 每个文件夹只跑前 6 张（先验证启发式）
"""
import json
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np

# ============ 路径 ============
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "training" / "photo"
OUTPUT_DIR = PROJECT_ROOT / "data" / "training" / "autolabel"
CLASS_MAP_PATH = PROJECT_ROOT / "data" / "yolo" / "class_map.json"

# 文件夹名(中文) -> 英文类别名（要和 class_map.json 的 key 一字不差）
# 注意：键名必须和 data/photo/ 下的实际文件夹名完全一致
FOLDER_TO_CLASS = {
    "数字万用表": "multimeter",
    "激光测距仪": "laser_meter",
    "绝缘钢丝钳": "pliers",
    "磁力线坠": "plumb_bob",
    "网线测距仪": "cable_tester",
    "卷尺": "ruler",
    "绝缘电阻测试仪": "insulation_tester",
    "胎压测试仪": "tire_gauge",
    "电工胶带": "tape",
}
# 多工具合照单独走 labelImg 人工标注（见 data/label_full），不在这里处理；
# 背景图(负样本)在 scripts/prepare_dataset.py 里接入。
MULTI_TOOL_FOLDER = None

# ============ 过滤参数 ============
MIN_AREA = 0.02     # 框面积下限（占整图）：低于这个当碎屑丢
MAX_AREA = 0.55     # 框面积上限：高于这个多半是把背景/桌面框进来了
# 背景块特征：横跨整幅宽度(或高度) + 另一维也很大 → 是桌面/地面，不是工具
BG_SPAN = 0.93      # 某一维 ≥ 此比例算"跨满"
BG_OTHER = 0.6      # 跨满 + 另一维 ≥ 此比例 → 判背景丢弃（长条工具如钳子另一维很小，不会误伤）
# 居中性权重（降低）：工具常不在正中，纹理才是主信号
CENTER_W = 0.3
# 多物体判定：次优框得分 ≥ 最优的这个比例，且两框中心离得远 → 疑似多工具图 → review
AMBIG_RATIO = 0.6
AMBIG_DIST = 0.25   # 两框中心归一化距离 > 此值才算"离得远"


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def imwrite(p, img):
    p = Path(p)
    cv2.imencode(p.suffix or ".jpg", img)[1].tofile(str(p))


def get_classes():
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        return list(json.load(f).keys())


def mask_to_box(mask, W, H):
    """单个 mask -> (area_frac, (x1,y1,x2,y2), (cxn,cyn)) ；空 mask 返回 None"""
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    frac = (x2 - x1) * (y2 - y1) / (W * H)
    cxn = (x1 + x2) / 2 / W
    cyn = (y1 + y2) / 2 / H
    return frac, (x1, y1, x2, y2), (cxn, cyn)


def pick_best(masks, grad, W, H):
    """从所有 mask 里挑"最像目标工具"的那个 + 判断是否可疑。

    打分 = 面积^0.5 × 纹理强度 × 居中性。纹理是主信号：光滑桌面梯度≈0，
    工具(按键/文字/标签)梯度高，所以哪怕工具偏离中心也能压过大片背景。

    Returns: (box or None, status)  status ∈ {"auto","review"}
    """
    cands = []
    for mk in masks:
        info = mask_to_box(mk, W, H)
        if info is None:
            continue
        frac, box, (cxn, cyn) = info
        if frac < MIN_AREA or frac > MAX_AREA:
            continue
        # 背景块过滤：跨满一维 + 另一维也大 → 桌面/地面
        x1, y1, x2, y2 = box
        wfrac = (x2 - x1) / W
        hfrac = (y2 - y1) / H
        if (wfrac >= BG_SPAN and hfrac >= BG_OTHER) or \
           (hfrac >= BG_SPAN and wfrac >= BG_OTHER):
            continue
        # 纹理：mask 内平均梯度（光滑桌面≈低，工具≈高）
        sel = mk > 0.5
        texture = float(grad[sel].mean()) if sel.any() else 0.0
        center_dist = ((cxn - 0.5) ** 2 + (cyn - 0.5) ** 2) ** 0.5  # 0~0.707
        score = (frac ** 0.5) * texture * (1 - CENTER_W * min(center_dist / 0.707, 1.0))
        cands.append((score, frac, box, (cxn, cyn)))

    if not cands:
        return None, "review"

    cands.sort(reverse=True)
    best = cands[0]
    # 多物体检测：是否存在另一个"够大、离得远"的强候选
    for other in cands[1:]:
        if other[0] >= AMBIG_RATIO * best[0]:
            d = ((other[3][0] - best[3][0]) ** 2 + (other[3][1] - best[3][1]) ** 2) ** 0.5
            if d > AMBIG_DIST:
                return best[2], "review"   # 疑似多工具 → 人工核对
    return best[2], "auto"


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else None  # None = 全部
    classes = get_classes()
    print(f"📋 {len(classes)} 类: {classes}")
    print(f"📂 输入: {SOURCE_DIR}")
    print(f"📂 输出: {OUTPUT_DIR}" + (f"（每文件夹仅前 {n_per} 张，验证模式）" if n_per else ""))

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"找不到输入目录: {SOURCE_DIR}")

    # 清空重建输出
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for sub in ("images", "labels", "preview"):
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    # 延迟 import（ultralytics 启动慢）
    print("\n📥 加载 FastSAM-s.pt ...")
    from ultralytics import FastSAM
    model = FastSAM(str(PROJECT_ROOT / "models" / "FastSAM-s.pt"))

    review_list = []
    stat = {"auto": 0, "review": 0, "multi": 0}

    def list_imgs(folder):
        return sorted([p for p in folder.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

    # ---- 单工具文件夹 ----
    for folder_name, cls_name in FOLDER_TO_CLASS.items():
        folder = SOURCE_DIR / folder_name
        if not folder.exists():
            print(f"  ⚠️  跳过不存在: {folder_name}")
            continue
        cls_id = classes.index(cls_name)
        imgs = list_imgs(folder)
        if n_per:
            imgs = imgs[:n_per]
        print(f"\n  [{cls_name}] {folder_name}/  {len(imgs)} 张")

        for i, p in enumerate(imgs):
            if i % 20 == 0 and i:
                print(f"    {i}/{len(imgs)}")
            img = imread(p)
            if img is None:
                continue
            H, W = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                                 cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
            r = model(img, retina_masks=True, conf=0.4, iou=0.9, verbose=False)[0]
            masks = r.masks.data.cpu().numpy() if r.masks is not None else []
            box, status = pick_best(masks, grad, W, H)

            stem = f"{cls_name}_{i:04d}"
            imwrite(OUTPUT_DIR / "images" / f"{stem}.jpg", img)

            preview = img.copy()
            if box is not None:
                x1, y1, x2, y2 = box
                xc = (x1 + x2) / 2 / W
                yc = (y1 + y2) / 2 / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                (OUTPUT_DIR / "labels" / f"{stem}.txt").write_text(
                    f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
                color = (0, 255, 0) if status == "auto" else (0, 165, 255)
                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 3)
            else:
                # 没框到：留空标签（labelImg 里手动补）
                (OUTPUT_DIR / "labels" / f"{stem}.txt").write_text("", encoding="utf-8")

            tag = "AUTO" if status == "auto" else "REVIEW"
            tcol = (0, 255, 0) if status == "auto" else (0, 165, 255)
            cv2.putText(preview, f"{tag} [{cls_name}]", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, tcol, 3)
            imwrite(OUTPUT_DIR / "preview" / f"{stem}.jpg", preview)

            stat[status] += 1
            if status == "review":
                review_list.append(f"{stem}.jpg  ({cls_name})")

    # ---- 多工具文件夹：全部进 review（类别无法自动定）----
    multi = (SOURCE_DIR / MULTI_TOOL_FOLDER) if MULTI_TOOL_FOLDER else None
    if multi and multi.exists():
        imgs = list_imgs(multi)
        if n_per:
            imgs = imgs[:n_per]
        print(f"\n  [多工具] {MULTI_TOOL_FOLDER}/  {len(imgs)} 张 → 全部需人工标注")
        for i, p in enumerate(imgs):
            img = imread(p)
            if img is None:
                continue
            stem = f"multi_{i:04d}"
            imwrite(OUTPUT_DIR / "images" / f"{stem}.jpg", img)
            (OUTPUT_DIR / "labels" / f"{stem}.txt").write_text("", encoding="utf-8")
            preview = img.copy()
            cv2.putText(preview, "REVIEW [multi-tool]", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            imwrite(OUTPUT_DIR / "preview" / f"{stem}.jpg", preview)
            stat["multi"] += 1
            review_list.append(f"{stem}.jpg  (多工具，需手动标全部框)")

    (OUTPUT_DIR / "review_list.txt").write_text(
        "\n".join(review_list) + "\n", encoding="utf-8")

    total = stat["auto"] + stat["review"] + stat["multi"]
    print(f"\n{'='*50}")
    print(f"✅ 完成 {total} 张")
    print(f"   🟢 AUTO   自动框且可信 : {stat['auto']}")
    print(f"   🟠 REVIEW 单工具但可疑 : {stat['review']}")
    print(f"   🔴 multi  多工具需全标 : {stat['multi']}")
    print(f"\n   下一步：用 labelImg 打开 {OUTPUT_DIR/'images'}（配 classes.txt，格式选 YOLO），")
    print(f"   先快速翻 {OUTPUT_DIR/'preview'} 找橙/红标记的重点改，绿的抽查即可。")
    print(f"   需核对清单见: {OUTPUT_DIR/'review_list.txt'}")


if __name__ == "__main__":
    main()
