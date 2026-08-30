r"""把 dist\智能工具检测系统\ 编译成一个 Setup.exe 安装程序（用 Inno Setup）。

完整出包流程（两步）：
    python scripts/build_exe.py          # 1. PyInstaller 打出 dist\智能工具检测系统\
    python scripts/build_installer.py    # 2. 本脚本：Inno Setup 把它封成 Setup.exe

产物：installer_output\智能工具检测系统_安装程序_v1.0.0.exe
    —— 这一个文件发给别人，双击即可安装到电脑（开始菜单/桌面快捷方式带图标，可卸载）。

前提：已安装 Inno Setup 6（winget install JRSoftware.InnoSetup）。
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ISS = PROJECT_ROOT / "installer.iss"
DIST = PROJECT_ROOT / "dist" / "智能工具检测系统"

# ISCC.exe 常见安装位置（winget 装的在用户目录，官网装的在 Program Files）
ISCC_CANDIDATES = [
    Path.home() / "AppData/Local/Programs/Inno Setup 6/ISCC.exe",
    Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe"),
    Path("C:/Program Files/Inno Setup 6/ISCC.exe"),
]


def find_iscc() -> Path:
    for p in ISCC_CANDIDATES:
        if p.is_file():
            return p
    print("❌ 找不到 Inno Setup 编译器 ISCC.exe。")
    print("   请先安装：winget install JRSoftware.InnoSetup")
    sys.exit(1)


def main() -> None:
    if not DIST.is_dir():
        print(f"❌ 未找到打包产物：{DIST}")
        print("   请先运行：python scripts/build_exe.py")
        sys.exit(1)
    if not (DIST / "智能工具检测系统.exe").is_file():
        print(f"❌ {DIST} 里没有 exe，产物不完整，请重新 build_exe.py")
        sys.exit(1)

    iscc = find_iscc()
    print("=" * 60)
    print("用 Inno Setup 编译安装程序……")
    print(f"  编译器: {iscc}")
    print(f"  脚本  : {ISS}")
    print("=" * 60)

    # ISCC 在 .iss 所在目录为工作目录解析相对路径，所以 cwd 设成项目根
    ret = subprocess.run([str(iscc), str(ISS)], cwd=str(PROJECT_ROOT))
    if ret.returncode != 0:
        print("\n❌ 编译失败（见上面 Inno Setup 输出）")
        sys.exit(ret.returncode)

    out_dir = PROJECT_ROOT / "installer_output"
    print("\n" + "=" * 60)
    print("✅ 安装程序生成完成！")
    for exe in out_dir.glob("*.exe"):
        print(f"   {exe}")
    print("把这个 Setup.exe 发给别人，双击即可安装到电脑。")
    print("=" * 60)


if __name__ == "__main__":
    main()
