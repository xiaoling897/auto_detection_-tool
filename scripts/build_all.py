r"""一键出包：PyInstaller 打包 + Inno Setup 封装 Setup.exe（两步合一）。

用法：
    python scripts/build_all.py              # 正式版（无黑窗）
    python scripts/build_all.py --console    # 调试版（保留黑窗看诊断）

等价于依次跑：
    python scripts/build_exe.py [--console]
    python scripts/build_installer.py

最终产物：installer_output\智能工具检测系统_安装程序_v1.0.0.exe
    —— 发这一个文件给别人，双击即可安装到电脑。
"""
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# 强制子进程用 UTF-8 输出：否则 Windows GBK 控制台下 build_exe.py 里的 emoji print
# 会 UnicodeEncodeError 直接中断出包。
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def run_step(title: str, script: str, extra=()) -> None:
    print("\n" + "#" * 60)
    print(f"# {title}")
    print("#" * 60)
    cmd = [PY, str(PROJECT_ROOT / "scripts" / script), *extra]
    ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=ENV)
    if ret.returncode != 0:
        print(f"\n[失败] 步骤 {script}（退出码 {ret.returncode}），已中止。")
        sys.exit(ret.returncode)


def main() -> None:
    console = ["--console"] if "--console" in sys.argv else []
    run_step("第 1 步 / 共 2 步：PyInstaller 打包", "build_exe.py", console)
    run_step("第 2 步 / 共 2 步：Inno Setup 封装 Setup.exe", "build_installer.py")

    print("\n" + "=" * 60)
    print("[完成] 全部搞定！安装程序在：")
    for exe in (PROJECT_ROOT / "installer_output").glob("*.exe"):
        print(f"   {exe}")
    print("发这一个文件给别人，双击即可安装到电脑。")
    print("=" * 60)


if __name__ == "__main__":
    main()
