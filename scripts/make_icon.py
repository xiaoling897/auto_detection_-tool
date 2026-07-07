"""生成应用图标 assets/app_icon.ico（工具检测主题）。

用法：
    python scripts/make_icon.py

想换成自己的图标有两种方式：
  1) 直接把你自己的 .ico 覆盖到 assets/app_icon.ico（打包脚本只认这个路径）；
  2) 或者把一张方形 png/jpg 传进来转换：
        python scripts/make_icon.py 你的图.png

打包时 scripts/build_exe.py 会自动读取 assets/app_icon.ico 作为 exe 图标，
Inno Setup 也用它做安装程序和快捷方式的图标。
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS = PROJECT_ROOT / "assets"
OUT = ASSETS / "app_icon.ico"

# 多尺寸：Windows 在任务栏/开始菜单/桌面/文件资源管理器用不同尺寸
SIZES = [256, 128, 64, 48, 32, 16]


def from_image(src: Path) -> Image.Image:
    """把用户给的图片裁成方形、缩放到 256。"""
    img = Image.open(src).convert("RGBA")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2,
                    (w - side) // 2 + side, (h - side) // 2 + side))
    return img.resize((256, 256), Image.LANCZOS)


def draw_default() -> Image.Image:
    """代码画一个「工具检测」主题图标：蓝色圆角底 + 白色扳手 + 检测取景框。"""
    S = 256
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # 圆角蓝色背景（竖向渐变，深蓝→亮蓝）
    radius = 52
    top = (37, 99, 235)     # #2563EB
    bot = (14, 165, 233)    # #0EA5E9
    bg = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bgd = ImageDraw.Draw(bg)
    for y in range(S):
        t = y / (S - 1)
        r = int(top[0] + (bot[0] - top[0]) * t)
        g = int(top[1] + (bot[1] - top[1]) * t)
        b = int(top[2] + (bot[2] - top[2]) * t)
        bgd.line([(0, y), (S, y)], fill=(r, g, b, 255))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=radius, fill=255)
    img.paste(bg, (0, 0), mask)
    d = ImageDraw.Draw(img)

    white = (255, 255, 255, 255)

    # 扳手：一根斜向的手柄 + 一端开口钳头（简化几何画法）
    # 手柄
    d.line([(96, 168), (168, 96)], fill=white, width=26)
    # 钳头（开口 C 形，用两段弧近似）
    d.arc([150, 60, 210, 120], start=200, end=110, fill=white, width=22)

    # 检测取景框（四个角标 L 形），表示「识别/框选」
    corner = 18
    off = 34
    cw = 12
    green = (190, 242, 100, 255)
    # 左上
    d.line([(off, off), (off + corner, off)], fill=green, width=cw)
    d.line([(off, off), (off, off + corner)], fill=green, width=cw)
    # 右下
    d.line([(S - off, S - off), (S - off - corner, S - off)], fill=green, width=cw)
    d.line([(S - off, S - off), (S - off, S - off - corner)], fill=green, width=cw)

    return img


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
        if not src.is_file():
            print(f"找不到图片：{src}")
            sys.exit(1)
        base = from_image(src)
        print(f"从图片生成：{src}")
    else:
        base = draw_default()
        print("生成默认工具主题图标")

    icons = [base.resize((s, s), Image.LANCZOS) for s in SIZES]
    icons[0].save(OUT, format="ICO", sizes=[(s, s) for s in SIZES])
    print(f"已写入：{OUT}")
    print("想换图标：把你自己的 .ico 覆盖这个文件，或 python scripts/make_icon.py 你的图.png")


if __name__ == "__main__":
    main()
