"""打包成可执行文件（onedir 文件夹分发，含 YOLO 模型）。

为什么是 onedir 而不是 onefile：
  依赖 torch + ultralytics 体积大，onefile 每次启动都要解压上百 MB、启动慢且易崩；
  onedir 把依赖摊在文件夹里，启动快、最稳。

产物：dist/智能工具检测系统/
        智能工具检测系统.exe
        _internal/...              （依赖，PyInstaller 生成）
        data/yolo/best.pt          （构建后复制进来，可随时替换换模型）
        data/yolo/class_map.json

在项目根目录运行：python scripts/build_exe.py
"""
import os
import shutil
import sys
from pathlib import Path

import PyInstaller.__main__

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# 加 --console 参数打"调试版"：保留黑窗，能看到 diag() 的实时诊断输出。
# 不加（默认）打正式版：--windowed 无黑窗。
DEBUG_CONSOLE = "--console" in sys.argv

APP_NAME = "智能工具检测系统"
DIST_DIR = PROJECT_ROOT / "dist" / APP_NAME

print("=" * 60)
print(f"打包：{APP_NAME}（onedir，{'调试版-带控制台' if DEBUG_CONSOLE else '正式版-无黑窗'}）")
print("=" * 60)

PyInstaller.__main__.run([
    "run.py",
    f"--name={APP_NAME}",
    # 调试版用 --console 看诊断输出；正式版 --windowed 不弹黑窗
    "--console" if DEBUG_CONSOLE else "--windowed",
    "--onedir",              # 文件夹分发（torch 重，onefile 不可取）
    "--noconfirm",
    "--clean",
    "--paths=src",
    # ultralytics 动态加载子模块 + 自带 cfg 数据文件，必须整包收集
    "--collect-all=ultralytics",
    # 关键 hidden import（GUI / 图像 / 语音）
    "--hidden-import=cv2",
    "--hidden-import=numpy",
    "--hidden-import=PIL",
    "--hidden-import=PIL._tkinter_finder",
    "--hidden-import=win32com",
    "--hidden-import=win32com.client",
])

# ============ 构建后：把 YOLO 模型外置到 exe 旁边 ============
# data/ 解析逻辑见 src/utils.py：frozen 时按 exe 同级目录找 data/
src_yolo = PROJECT_ROOT / "data" / "yolo"
dst_yolo = DIST_DIR / "data" / "yolo"
if src_yolo.is_dir():
    dst_yolo.mkdir(parents=True, exist_ok=True)
    for fn in ("best.pt", "class_map.json"):
        f = src_yolo / fn
        if f.is_file():
            shutil.copy2(f, dst_yolo / fn)
            print(f"  已复制模型文件: {fn}")
        else:
            print(f"  缺少 {fn}（best.pt 缺失则程序启动后点检测会提示）")
else:
    print("  data/yolo 不存在，未复制模型")

print("\n" + "=" * 60)
print("打包完成！")
print(f"可执行文件：{DIST_DIR / (APP_NAME + '.exe')}")
print(f"整个 {DIST_DIR.name}/ 文件夹拷给别人即可双击运行。")
print("=" * 60)
