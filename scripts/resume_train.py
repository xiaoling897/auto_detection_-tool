"""从断点续训 YOLO 工具检测模型。

用途：之前 train_yolo.py 跑到一半（电脑故障/手动中断）后，从 last.pt 接着跑到 80 epoch，
不用从头开始。ultralytics 会从 checkpoint 里读回原始训练参数（epochs/batch/增强等），
自动从中断的那一轮继续。

完成后和 train_yolo.py 一样：把 best.pt 复制到 data/yolo/best.pt。
"""
import os

# ⚠️ 必须在 import torch/ultralytics 之前设：限制 OpenMP/MKL 线程数。
# 这台机器有 28 个逻辑核，torch 默认会铺满所有核跑 OpenMP，hybrid P/E 核 + 高线程
# 在 Windows 上长时间满载会偶发原生层崩溃（无 Python 堆栈、无 Windows 崩溃事件、
# 训练跑几分钟后从迭代中途硬断）——之前三次训练都死在这上面。把线程数压到 8 既稳又够快。
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 顺手防 libiomp 重复初始化 abort

# ⚠️ 让训练进程完全不持有任何 GPU/图形资源。
# 这台机器的显示驱动（向日葵虚拟显示器 OrayIddDriver / Intel UHD 770 旧驱动）在高负载下
# 会偶发 TDR 崩溃（事件日志里 0x116/0x117/0x141 video TDR），把"摸过 GPU"的进程一起杀掉。
# torch 会探测 CUDA、cv2 默认开 OpenCL —— 都会拿到图形句柄而被 TDR 连坐。
# 训练本就是纯 CPU，这里强制屏蔽 GPU，进程不再有图形句柄，显示驱动崩了也波及不到它。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")       # torch 看不到任何 CUDA 设备
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")  # cv2 不走 OpenCL/GPU
os.environ.setdefault("OPENCV_OPENCL_DEVICE", "disabled")

import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

torch.set_num_threads(8)

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
