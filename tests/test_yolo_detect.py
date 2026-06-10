"""命令行测试：用 YOLO 后端对 data/samples 下图片做检测，输出每张命中数。

用法：
    python tests/test_yolo_detect.py                 # 测 data/samples 全部
    python tests/test_yolo_detect.py <图片路径>       # 测指定图
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import DATA_DIR, imread_unicode
from src.yolo_detector import YoloToolDetector

sys.stdout.reconfigure(encoding='utf-8')


def main():
    model = DATA_DIR / "yolo" / "best.pt"
    cmap = DATA_DIR / "yolo" / "class_map.json"
    engine = YoloToolDetector(str(model), str(cmap))
    total_classes = len(engine.model.names)
    print(f"YOLO 模型类别数: {total_classes} -> {list(engine.model.names.values())}\n")

    if len(sys.argv) > 1:
        imgs = [Path(a) for a in sys.argv[1:]]
    else:
        folder = DATA_DIR / "samples"
        imgs = sorted(p for p in folder.iterdir()
                      if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))

    for p in imgs:
        frame = imread_unicode(str(p))
        if frame is None:
            print(f"读取失败: {p}")
            continue
        h, w = frame.shape[:2]
        detected, _display, counts = engine.detect(frame)
        print(f"=== {p.name} ({w}x{h}) ===")
        for name, conf in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  [✓] {name:12s} 置信度={conf:.2f}")
        print(f"  检测到 {len(detected)}/{total_classes}: {detected}\n")


if __name__ == "__main__":
    main()
