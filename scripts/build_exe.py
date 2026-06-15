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

# ============ 构建后：把运行时数据外置到 exe 旁边 ============
# data/ 解析逻辑见 src/utils.py：frozen 时按 exe 同级目录找 data/。
# YOLO 模式运行时只需要 yolo/best.pt + yolo/class_map.json；
# smart_tools.json 仅在 best.pt 缺失回退 SIFT 时才用到，一并带上做兜底。
dst_data = DIST_DIR / "data"

# 1) YOLO 模型 + 类名映射（核心）
src_yolo = PROJECT_ROOT / "data" / "yolo"
dst_yolo = dst_data / "yolo"
if src_yolo.is_dir():
    dst_yolo.mkdir(parents=True, exist_ok=True)
    for fn in ("best.pt", "class_map.json"):
        f = src_yolo / fn
        if f.is_file():
            shutil.copy2(f, dst_yolo / fn)
            print(f"  已复制: data/yolo/{fn}")
        else:
            print(f"  ⚠️ 缺少 data/yolo/{fn}（best.pt 缺失则启动后点检测会提示）")
else:
    print("  ⚠️ data/yolo 不存在，未复制模型")

# 2) smart_tools.json（SIFT 回退兜底，可选）
sj = PROJECT_ROOT / "data" / "smart_tools.json"
if sj.is_file():
    dst_data.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sj, dst_data / "smart_tools.json")
    print("  已复制: data/smart_tools.json（回退兜底）")

# 3) 选图/选视频对话框的默认目录（建空目录，避免对话框打开到奇怪位置）
for sub in ("samples", "video"):
    src_sub = PROJECT_ROOT / "data" / sub
    dst_sub = dst_data / sub
    if src_sub.is_dir():
        shutil.copytree(src_sub, dst_sub, dirs_exist_ok=True)
        print(f"  已复制: data/{sub}/")
    else:
        dst_sub.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 60)
print("✅ 打包完成！")
print(f"双击运行：{DIST_DIR / (APP_NAME + '.exe')}")
print(f"把整个『{DIST_DIR.name}』文件夹拷给别人，双击里面的 .exe 即可秒开，无需任何命令。")
print("=" * 60)
