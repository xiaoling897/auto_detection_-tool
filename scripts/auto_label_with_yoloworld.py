"""用 YOLO-World 零样本检测自动标注工具数据集，输出 YOLO 训练用的标签 + data.yaml。

数据集结构（输入）：
  C:/Users/acad1/Desktop/tool_dataset/
    万用表/          ← 单工具文件夹，文件夹名 = 类别
    激兆测距仪/
    ...（9 个）
    全部在一起照片/  ← 多工具文件夹

输出（项目内）：
  data/yolo_dataset/
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt   ← YOLO 格式: class_id xc yc w h（归一化）
    preview/{train,val,test}/*.jpg  ← 标了框的预览图，让你抽查 AI 标得准不准
    data.yaml                       ← 训练配置
"""
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOWorld

# ============ 路径配置 ============
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SOURCE_DIR = Path(r"C:\Users\acad1\Desktop\tool_dataset")
OUTPUT_DIR = PROJECT_ROOT / "data" / "yolo_dataset"
CLASS_MAP_PATH = PROJECT_ROOT / "data" / "yolo" / "class_map.json"

# ============ 工具映射 ============
# 文件夹名（中文） -> 英文类别名（要和 class_map.json 的 key 一字不差）
FOLDER_TO_CLASS = {
    "万用表": "multimeter",
    "激兆测距仪": "laser_meter",
    "绝缘钢丝钳": "pliers",
    "磁力线坠": "plumb_bob",
    "网线测线仪": "cable_tester",
    "卷尺": "ruler",
    "绝缘测试仪": "insulation_tester",
    "胎压测试仪": "tire_gauge",
    "电工胶带": "tape",
}
MULTI_TOOL_FOLDER = "全部在一起照片"

# YOLO-World 是英文 vision-language 模型，给它更具体的英文描述帮它理解
CLASS_PROMPTS = {
    "multimeter": "digital multimeter",
    "laser_meter": "laser distance meter",
    "pliers": "wire cutter pliers",
    "plumb_bob": "metal plumb bob",
    "cable_tester": "network cable tester",
    "ruler": "yellow tape measure",
    "insulation_tester": "insulation resistance tester",
    "tire_gauge": "tire pressure gauge",
    "tape": "electrical tape roll",
}

# ============ 训练参数 ============
SPLIT_RATIOS = (0.7, 0.2, 0.1)      # train / val / test
CONF_THRESHOLD = 0.05               # YOLO-World 置信度阈值（低一点，靠面积过滤）
MIN_BOX_AREA_RATIO = 0.02           # 框面积至少占整张图 2%，过滤噪声小框

random.seed(42)


# ============ 工具函数 ============
def imread_unicode(path):
    """读中文路径下的图片（cv2.imread 在 Windows 中文路径下静默失败）"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img):
    """写中文路径（虽然这里输出路径不含中文，但保持一致）"""
    ext = path.suffix or ".jpg"
    success, buf = cv2.imencode(ext, img)
    if not success:
        return False
    buf.tofile(str(path))
    return True


def get_class_names():
    """从 class_map.json 读取英文类别名，顺序就是 JSON 文件里的 key 顺序"""
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return list(mapping.keys())


def make_dirs():
    for split in ("train", "val", "test"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "preview" / split).mkdir(parents=True, exist_ok=True)


def detect(model, img_array, prompts):
    """跑一次 YOLO-World 检测，返回 [(prompt_idx, xyxy, conf), ...]"""
    model.set_classes(prompts)
    results = model.predict(img_array, conf=CONF_THRESHOLD, verbose=False)
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            xyxy = box.xyxy[0].cpu().numpy()
            detections.append((cls, xyxy, conf))
    return detections


def xyxy_to_yolo(xyxy, img_w, img_h):
    """(x1, y1, x2, y2) 绝对坐标 → (xc, yc, w, h) 归一化"""
    x1, y1, x2, y2 = xyxy
    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return max(0, xc), max(0, yc), min(1, w), min(1, h)


# ============ 处理函数 ============
def process_single_tool_folder(model, folder_path, class_name, classes):
    """单工具文件夹：用专属 prompt 找出工具，找不到就用中心 80% 作为框（fallback）"""
    class_idx = classes.index(class_name)
    prompt = CLASS_PROMPTS[class_name]
    print(f"\n  [{class_name}] 处理 {folder_path.name}/")

    imgs = sorted([p for p in folder_path.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    results = []
    detected_count = 0
    fallback_count = 0

    for i, img_path in enumerate(imgs):
        if i % 20 == 0:
            print(f"    {i}/{len(imgs)}")
        img = imread_unicode(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        detections = detect(model, img, [prompt])

        valid = []
        for _, xyxy, conf in detections:
            bw = xyxy[2] - xyxy[0]
            bh = xyxy[3] - xyxy[1]
            area_ratio = (bw * bh) / (w * h)
            if area_ratio >= MIN_BOX_AREA_RATIO:
                valid.append((xyxy, conf, area_ratio))

        if valid:
            # 取面积最大的检测（最显眼的工具）
            valid.sort(key=lambda v: v[2], reverse=True)
            xyxy, _, _ = valid[0]
            detected_count += 1
        else:
            # 兜底：中心 80% 区域作为框（单工具特写时一般工具就在中间）
            cx, cy = w / 2, h / 2
            xyxy = (cx - 0.4 * w, cy - 0.4 * h, cx + 0.4 * w, cy + 0.4 * h)
            fallback_count += 1

        xc, yc, bw, bh = xyxy_to_yolo(xyxy, w, h)
        label_text = f"{class_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
        results.append((img_path, img, label_text, [(class_idx, xyxy)]))

    print(f"    ✓ {len(results)} 张完成 (AI检测: {detected_count}, 兜底中心框: {fallback_count})")
    return results


def process_multi_tool_folder(model, folder_path, classes):
    """多工具文件夹：用 9 个 prompt 一起检测，每张图可能有多个框"""
    prompts = [CLASS_PROMPTS[c] for c in classes]
    print(f"\n  [多工具] 处理 {folder_path.name}/")

    imgs = sorted([p for p in folder_path.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    results = []
    skipped = 0

    for i, img_path in enumerate(imgs):
        if i % 10 == 0:
            print(f"    {i}/{len(imgs)}")
        img = imread_unicode(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        detections = detect(model, img, prompts)

        label_lines = []
        boxes_for_preview = []
        for cls, xyxy, conf in detections:
            bw = xyxy[2] - xyxy[0]
            bh = xyxy[3] - xyxy[1]
            if (bw * bh) / (w * h) < MIN_BOX_AREA_RATIO:
                continue
            xc, yc, bbw, bbh = xyxy_to_yolo(xyxy, w, h)
            label_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bbw:.6f} {bbh:.6f}")
            boxes_for_preview.append((cls, xyxy))

        if not label_lines:
            skipped += 1
            continue

        label_text = "\n".join(label_lines) + "\n"
        results.append((img_path, img, label_text, boxes_for_preview))

    print(f"    ✓ {len(results)} 张完成 (跳过 {skipped} 张未检测到工具的)")
    return results


# ============ 保存 + 划分 ============
def split_and_save(all_results, classes):
    """打乱 + 70/20/10 划分 + 保存图片/标签/预览"""
    random.shuffle(all_results)
    n = len(all_results)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])

    splits = {
        "train": all_results[:n_train],
        "val": all_results[n_train:n_train + n_val],
        "test": all_results[n_train + n_val:],
    }

    print(f"\n📊 数据集划分:")
    for split, items in splits.items():
        print(f"  {split}: {len(items)} 张")

    for split, items in splits.items():
        for i, (img_path, img, label_text, boxes) in enumerate(items):
            stem = f"{img_path.stem}_{i:04d}"

            # 图片（统一存 .jpg）
            out_img = OUTPUT_DIR / "images" / split / f"{stem}.jpg"
            imwrite_unicode(out_img, img)

            # 标签
            out_lbl = OUTPUT_DIR / "labels" / split / f"{stem}.txt"
            with open(out_lbl, "w", encoding="utf-8") as f:
                f.write(label_text)

            # 预览图（画框）
            preview = img.copy()
            for cls_idx, xyxy in boxes:
                x1, y1, x2, y2 = map(int, xyxy)
                color = (0, 255, 0)
                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
                label_str = classes[cls_idx]
                cv2.putText(preview, label_str, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            imwrite_unicode(OUTPUT_DIR / "preview" / split / f"{stem}.jpg", preview)


def write_data_yaml(classes):
    yaml_text = (
        "# Auto-generated by scripts/auto_label_with_yoloworld.py\n"
        f"path: {OUTPUT_DIR.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )
    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"\n✅ data.yaml 已生成: {yaml_path}")


# ============ 主流程 ============
def main():
    print("🤖 YOLO-World 自动标注开始")
    print(f"📁 输入: {SOURCE_DIR}")
    print(f"📁 输出: {OUTPUT_DIR}")

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"找不到输入目录: {SOURCE_DIR}")

    classes = get_class_names()
    print(f"\n📋 {len(classes)} 个类别: {classes}")

    # 清空 + 重建输出目录
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    make_dirs()

    # 加载 YOLO-World（首次会下载 ~25MB）
    print("\n📥 加载 YOLO-World 模型...")
    model = YOLOWorld("yolov8s-worldv2.pt")

    all_results = []

    print("\n🔍 单工具文件夹（用专属 prompt）:")
    for folder_name, class_name in FOLDER_TO_CLASS.items():
        folder = SOURCE_DIR / folder_name
        if not folder.exists():
            print(f"  ⚠️  跳过不存在: {folder_name}")
            continue
        all_results.extend(process_single_tool_folder(model, folder, class_name, classes))

    print("\n🔍 多工具文件夹（9 个 prompt 一起跑）:")
    multi_folder = SOURCE_DIR / MULTI_TOOL_FOLDER
    if multi_folder.exists():
        all_results.extend(process_multi_tool_folder(model, multi_folder, classes))

    print(f"\n✅ 标注完成: 共 {len(all_results)} 张有效图")

    split_and_save(all_results, classes)
    write_data_yaml(classes)

    print(f"\n🎉 全部完成！")
    print(f"   ▸ 检查预览图: {OUTPUT_DIR / 'preview'}")
    print(f"   ▸ 训练命令: python scripts/train_yolo.py")


if __name__ == "__main__":
    main()
