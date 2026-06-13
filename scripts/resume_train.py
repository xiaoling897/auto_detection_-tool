"""从断点续训 YOLO 工具检测模型。

用途：之前 train_yolo.py 跑到一半（电脑故障/手动中断）后，从 last.pt 接着跑到 80 epoch，
不用从头开始。ultralytics 会从 checkpoint 里读回原始训练参数（epochs/batch/增强等），
自动从中断的那一轮继续。

完成后和 train_yolo.py 一样：把 best.pt 复制到 data/yolo/best.pt。
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUNS_DIR = PROJECT_ROOT / "data" / "training" / "runs" / "tool_yolo"
LAST_PT = RUNS_DIR / "weights" / "last.pt"
BEST_PT = RUNS_DIR / "weights" / "best.pt"
OUTPUT_BEST = PROJECT_ROOT / "data" / "yolo" / "best.pt"


def main():
    if not LAST_PT.exists():
        raise FileNotFoundError(f"找不到断点权重 {LAST_PT}，无法续训")

    print(f"🔄 从断点续训: {LAST_PT}")
    model = YOLO(str(LAST_PT))
    model.train(resume=True)

    if BEST_PT.exists():
        OUTPUT_BEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(BEST_PT, OUTPUT_BEST)
        print(f"\n✅ 续训完成！最佳模型已复制到: {OUTPUT_BEST}")
        print(f"   原始位置: {BEST_PT}")
    elif LAST_PT.exists():
        OUTPUT_BEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(LAST_PT, OUTPUT_BEST)
        print(f"\n⚠️  best.pt 不存在，用了 last.pt: {OUTPUT_BEST}")
    else:
        print(f"\n❌ 续训完成但找不到 best.pt/last.pt，请检查: {RUNS_DIR}/weights/")

    print("\n🎉 现在可以运行 python run.py 看效果")


if __name__ == "__main__":
    main()
