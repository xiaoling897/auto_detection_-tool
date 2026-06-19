"""看门狗：反复续训直到跑满目标 epoch，自动扛住 torch 原生层偶发崩溃。

背景：这台机器 28 逻辑核，torch 原生层（OpenMP）长时间满载会偶发崩溃——无 Python 堆栈、
无 Windows 崩溃事件，训练跑十几个 epoch 就被硬断一次（已把线程数压到 8，崩溃变稀但没根除）。
关键是 ultralytics 每个 epoch 都存 last.pt，所以崩了就从 last.pt 续训，磨也能磨完。
本脚本就是这个「崩了自动重启续训」的循环，跑满 80 epoch 才停。

用法：python scripts/train_until_done.py
（建议用 Start-Process 脱离 shell 后台跑，避免 shell 会话回收。）
"""
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "training" / "runs" / "tool_yolo"
RESULTS = RUNS / "results.csv"
BEST_PT = RUNS / "weights" / "best.pt"
RESUME = ROOT / "scripts" / "resume_train.py"
OUTPUT_BEST = ROOT / "data" / "yolo" / "best.pt"
STATUS_LOG = ROOT / "data" / "training" / "watchdog_status.log"
TARGET_EPOCHS = 80
MAX_RESTARTS = 60


def log(msg):
    """状态写到独立 utf-8 日志文件，不依赖 stdout——这样即使被 WMI 无控制台启动
    （没有可重定向的 stdout）也能看到进度，且不会因 emoji 在 gbk 控制台报 UnicodeEncodeError。"""
    line = f"[{int(time.time())}] {msg}"
    try:
        with open(STATUS_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    try:
        print(line, flush=True)
    except Exception:
        pass


def last_epoch():
    """results.csv 最后一行的 epoch 编号（1 起算）；没有则 0。"""
    if not RESULTS.exists():
        return 0
    try:
        rows = list(csv.reader(open(RESULTS, encoding="utf-8")))
        if len(rows) <= 1:
            return 0
        return int(rows[-1][0])
    except Exception:
        return 0


def deploy_best():
    if BEST_PT.exists():
        OUTPUT_BEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(BEST_PT, OUTPUT_BEST)
        print(f"✅ 已部署最佳模型 → {OUTPUT_BEST}")


def main():
    log(f"看门狗启动，目标 {TARGET_EPOCHS} epoch。")
    for attempt in range(1, MAX_RESTARTS + 1):
        ep = last_epoch()
        if ep >= TARGET_EPOCHS:
            log(f"已完成 epoch {ep}/{TARGET_EPOCHS}，训练结束。")
            deploy_best()
            return
        log(f"[第 {attempt} 次] 已完成 epoch {ep}/{TARGET_EPOCHS}，启动续训...")

        # 子进程输出收到独立文件，避免无控制台时写丢；也方便排查偶发崩溃。
        run_log = ROOT / "data" / "training" / f"resume_run_{attempt}.log"
        with open(run_log, "w", encoding="utf-8", errors="replace") as f:
            rc = subprocess.run([sys.executable, str(RESUME)],
                                stdout=f, stderr=subprocess.STDOUT).returncode
        ep2 = last_epoch()

        if ep2 >= TARGET_EPOCHS:
            log(f"训练跑满 {ep2} epoch，完成（最后一次 rc={rc}）。")
            deploy_best()
            return
        if ep2 <= ep:
            # 没推进：resume 可能一启动就失败（环境/路径问题），别空转，等几秒再试
            log(f"本次没推进（epoch {ep}→{ep2}, rc={rc}）。等 8s 重试。")
            time.sleep(8)
        else:
            log(f"推进到 epoch {ep2}（rc={rc}），疑似又被打断，自动续训。")

    log(f"已重启 {MAX_RESTARTS} 次仍未跑满 {TARGET_EPOCHS}，停止，请人工检查。")
    deploy_best()


if __name__ == "__main__":
    main()
