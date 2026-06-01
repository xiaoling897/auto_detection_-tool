"""训练 YOLOv8n 工具检测模型（CPU 友好设置）。

依赖 scripts/auto_label_with_yoloworld.py 先跑过，生成了 data/yolo_dataset/。

训练完成后：自动把 best.pt 复制到 data/yolo/best.pt，覆盖之前的占位模型。
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

# ============ 路径 ============
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_YAML = PROJECT_ROOT / "data" / "yolo_dataset" / "data.yaml"
OUTPUT_BEST = PROJECT_ROOT / "data" / "yolo" / "best.pt"

# ============ 训练参数 ============
# CPU 训练优化：小图、小 batch、少 epoch
# GPU 用户可以提到 epochs=100, imgsz=640, batch=16
TRAIN_CONFIG = dict(
    data=str(DATA_YAML),
    model="yolov8n.pt",     # nano，最小最快，6MB
    epochs=50,              # 50 个 epoch 对 700 张图够用
    imgsz=416,              # 比默认 640 小，CPU 友好
    batch=8,                # CPU 友好的 batch
    workers=2,              # CPU 多线程数据加载
    device="cpu",
    project=str(PROJECT_ROOT / "data" / "runs"),  # 训练日志/权重保存位置
    name="tool_yolo",       # 子目录名
    exist_ok=True,          # 覆盖已有目录
    patience=15,            # 验证集 15 个 epoch 没提升就早停
    save=True,
    save_period=10,         # 每 10 epoch 存一次 checkpoint
    plots=True,
    verbose=True,
)


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"找不到 {DATA_YAML}\n请先运行: python scripts/auto_label_with_yoloworld.py"
        )

    print("🚀 开始训练 YOLOv8n")
    print(f"📋 数据集: {DATA_YAML}")
    print(f"⚙️  配置: epochs={TRAIN_CONFIG['epochs']}, "
          f"imgsz={TRAIN_CONFIG['imgsz']}, batch={TRAIN_CONFIG['batch']}, device=cpu")
    print(f"⏱️  CPU 训练约需 2-6 小时（看数据量和 CPU 性能），请耐心等待\n")

    model = YOLO(TRAIN_CONFIG["model"])
    results = model.train(**{k: v for k, v in TRAIN_CONFIG.items() if k != "model"})

    # 训练完产物：data/runs/tool_yolo/weights/best.pt
    runs_dir = Path(TRAIN_CONFIG["project"]) / TRAIN_CONFIG["name"]
    best_pt = runs_dir / "weights" / "best.pt"
    last_pt = runs_dir / "weights" / "last.pt"

    if best_pt.exists():
        OUTPUT_BEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, OUTPUT_BEST)
        print(f"\n✅ 训练完成！最佳模型已复制到: {OUTPUT_BEST}")
        print(f"   原始位置: {best_pt}")
    elif last_pt.exists():
        OUTPUT_BEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(last_pt, OUTPUT_BEST)
        print(f"\n⚠️  best.pt 不存在，用了 last.pt: {OUTPUT_BEST}")
    else:
        print(f"\n❌ 训练完成但找不到 best.pt 或 last.pt，请检查: {runs_dir}/weights/")

    print("\n🎉 现在可以运行 python run.py 看效果")


if __name__ == "__main__":
    main()
