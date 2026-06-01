"""打包成单文件 EXE。在项目根目录运行：python scripts/build_exe.py"""
import os
from pathlib import Path

import PyInstaller.__main__

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

print("=" * 60)
print("AI Tool Detection System - Build Script")
print("=" * 60)

PyInstaller.__main__.run([
    'run.py',
    '--name=AI_Tool_Detection_System',
    '--windowed',
    '--onefile',
    '--icon=NONE',
    '--add-data=data/smart_tools.json;data',
    '--add-data=data/smart_templates;data/smart_templates',
    '--add-data=data/templates;data/templates',
    '--paths=src',
    '--hidden-import=cv2',
    '--hidden-import=numpy',
    '--hidden-import=PIL',
    '--hidden-import=PIL._tkinter_finder',
    '--hidden-import=win32com',
    '--hidden-import=win32com.client',
    '--clean',
    '--noconfirm',
])

print("\n" + "=" * 60)
print("Build Complete!")
print("Executable: dist/AI_Tool_Detection_System.exe")
print("=" * 60)
