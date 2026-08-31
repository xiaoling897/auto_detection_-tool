"""公用工具：路径解析、中文路径读图"""
import sys
from pathlib import Path
import cv2
import numpy as np

# 打包成 EXE（PyInstaller，sys.frozen）后，data/ 放在 exe 同级目录；
# 普通源码运行时按项目根定位。两种情况都解析到一个 data/ 目录。
if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """读取包含中文路径的图片（cv2.imread 在 Windows + 中文路径下会失败）"""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, flags)


def resolve_data_path(rel_path) -> Path:
    """smart_tools.json 里的 template_path 是相对路径，统一解析到 data/ 下"""
    normalized = str(rel_path).replace('\\', '/')
    return DATA_DIR / normalized
