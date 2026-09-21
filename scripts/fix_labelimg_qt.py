"""修 labelImg 在新版 PyQt5 上画框时崩溃的 bug（TypeError: unexpected type 'float'）。

症状：按 W 进入画框模式、鼠标一动就崩，报
    File ".../libs/canvas.py", line 530, in paintEvent
      p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())
    TypeError: arguments did not match any overloaded call

原因：PyQt5 5.13 以后收紧了类型检查，drawLine()/drawRect() 只收整数坐标，
而 QPointF.x() 返回的是浮点数。老版本会自动取整，新版本直接抛 TypeError。

修法：给这几处坐标参数套上 int()。本脚本自动定位 labelImg 的安装位置并改，
改之前会存一份 canvas.py.bak。已经修过的会直接跳过，重复运行无害。

用法：  python scripts/fix_labelimg_qt.py
"""
import shutil
import sys
from pathlib import Path

# (原文, 改成) —— 都是整行匹配，避免误伤
FIXES = [
    ("p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())",
     "p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), int(self.pixmap.height()))"),
    ("p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())",
     "p.drawLine(0, int(self.prev_point.y()), int(self.pixmap.width()), int(self.prev_point.y()))"),
    ("p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)",
     "p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))"),
]


def find_canvas():
    """定位 labelImg 的 libs/canvas.py。"""
    try:
        import libs.canvas
        return Path(libs.canvas.__file__)
    except Exception:
        pass
    # import 失败（比如缺 PyQt5）时按路径找
    for base in sys.path:
        p = Path(base) / "libs" / "canvas.py"
        if p.is_file():
            return p
    return None


def main():
    canvas = find_canvas()
    if canvas is None:
        print("❌ 找不到 labelImg 的 libs/canvas.py。")
        print("   先确认装了：pip install labelImg")
        return 1

    print(f"📄 目标文件：{canvas}")
    text = canvas.read_text(encoding="utf-8")

    todo = [(old, new) for old, new in FIXES if old in text]
    already = [old for old, new in FIXES if old not in text and new in text]

    if not todo:
        if already:
            print(f"✅ 已经是修好的版本（{len(already)}/{len(FIXES)} 处都带 int()），不用再改。")
        else:
            print("⚠️  没找到要改的代码，也没找到已修过的痕迹。")
            print("   可能是 labelImg 版本不同。把崩溃时的报错行发我，我按你的版本改。")
        return 0

    bak = canvas.with_suffix(".py.bak")
    if not bak.exists():
        shutil.copyfile(canvas, bak)
        print(f"💾 已备份：{bak}")

    for old, new in todo:
        text = text.replace(old, new)
    canvas.write_text(text, encoding="utf-8")

    print(f"🔧 已修复 {len(todo)} 处")
    for old, _ in todo:
        print(f"   - {old[:60]}...")
    print("\n✅ 完成。重新启动标注软件即可：")
    print("   python scripts/open_labelimg.py <批次名>")
    print(f"\n（想还原：把 {bak.name} 覆盖回 {canvas.name}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
