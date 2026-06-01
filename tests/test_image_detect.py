"""命令行测试：使用 src.loader / src.detector 对图片做检测

用法：
    python tests/test_image_detect.py                  # 测 data/samples/ 下所有图
    python tests/test_image_detect.py <图片路径>        # 测指定图
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.detector import INLIER_THRESHOLD, ToolDetector
from src.loader import DEFAULT_NFEATURES, load_tools
from src.utils import DATA_DIR, imread_unicode

sys.stdout.reconfigure(encoding='utf-8')


def main():
    sift = cv2.SIFT_create(nfeatures=DEFAULT_NFEATURES)
    tools = load_tools(detector=sift)
    engine = ToolDetector(tools, detector=sift)

    valid = sum(1 for t in tools if t.get("des") is not None)
    print()

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
        for name, cnt in counts.items():
            mark = "✓" if cnt >= INLIER_THRESHOLD else " "
            print(f"  [{mark}] {name:15s} 内点={cnt:3d}")
        print(f"  检测到 {len(detected)}/{valid}: {detected}\n")


if __name__ == "__main__":
    main()
