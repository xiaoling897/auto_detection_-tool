"""一次性修好 labelImg 1.8.6 在本项目环境下的全部问题（12 处，改完就能正常用）。

labelImg 2021 年后就没更新过，在新版 PyQt5 + 中文 Windows 上有三类毛病：

  A. 画框/滚轮/缩放就崩（8 处）
     PyQt5 5.13 以后收紧类型检查，drawLine/drawRect/drawText/setValue 只收整数，
     而 labelImg 传的是 QPointF.x() 之类的浮点数。老 PyQt5 自动取整，新的直接抛
     TypeError: argument 1 has unexpected type 'float'。
     这些地方是分散的：按 W 画框崩一次，滚轮滚动崩一次，缩放又崩一次——
     所以必须一次全改，一处一处修会没完没了。

  B. 中文类别名乱码（3 处）
     libs/yolo_io.py 读写 classes.txt 没指定编码，中文 Windows 上默认走 GBK，
     而本项目的脚本一律按 UTF-8 读，会读出乱码或直接报错。

  C. 界面是英文（1 处）
     libs/stringBundle.py 靠 locale.getlocale() 认语言，中文 Windows 上返回
     'Chinese (Simplified)_China'，按 [^a-zA-Z] 切出来是 ':/strings-Chinese'，
     而内置资源里只有 ':/strings-zh-CN'，匹配不上就退回英文。

这些改动是对照 PyPI 上的原始 labelImg-1.8.6 源码逐行 diff 出来的，不是猜的。

用法：  python scripts/fix_labelimg.py
每个被改的文件都会先存一份 .bak。已经修过的会跳过，重复运行无害。
还原：把对应的 .bak 覆盖回去。
"""
import shutil
import sys
from pathlib import Path

# 文件 -> [(原文, 改成), ...]；都是整行唯一匹配，不会误伤
PATCHES = {
    "labelImg.py": [
        # A. 滚轮滚动
        ("bar.setValue(bar.value() + bar.singleStep() * units)",
         "bar.setValue(int(bar.value() + bar.singleStep() * units))"),
        # A. 设置缩放
        ("self.zoom_widget.setValue(value)",
         "self.zoom_widget.setValue(int(value))"),
        # A. 缩放时同步滚动条
        ("h_bar.setValue(new_h_bar_value)",
         "h_bar.setValue(int(new_h_bar_value))"),
        ("v_bar.setValue(new_v_bar_value)",
         "v_bar.setValue(int(new_v_bar_value))"),
    ],
    "libs/canvas.py": [
        # A. 画框时的矩形预览
        ("p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)",
         "p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))"),
        # A. 画框模式的十字准线（按 W 之后一动鼠标就崩的就是这两行）
        ("p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())",
         "p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), int(self.pixmap.height()))"),
        ("p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())",
         "p.drawLine(0, int(self.prev_point.y()), int(self.pixmap.width()), int(self.prev_point.y()))"),
    ],
    "libs/shape.py": [
        # A. 在框上画类别文字
        ("painter.drawText(min_x, min_y, self.label)",
         "painter.drawText(int(min_x), int(min_y), self.label)"),
    ],
    "libs/yolo_io.py": [
        # B. classes.txt 的读写编码（中文类名必须 UTF-8）
        ("            out_class_file = open(classes_file, 'w')\n",
         "            out_class_file = open(classes_file, 'w', encoding='utf-8')\n"),
        ("        classes_file = open(self.class_list_path, 'r')\n",
         "        classes_file = open(self.class_list_path, 'r', encoding='utf-8')\n"),
    ],
    "libs/stringBundle.py": [
        # C. 界面语言
        ("        return StringBundle(cls.__create_key, locale_str)",
         "        # [本项目修改] Windows 风格的中文 locale 名归一成 zh_CN / zh_TW，\n"
         "        # 否则匹配不到内置的 ':/strings-zh-CN'，界面会退回英文。\n"
         "        if locale_str:\n"
         "            _l = str(locale_str).lower()\n"
         "            if 'chinese' in _l or _l.startswith('zh'):\n"
         "                locale_str = 'zh_TW' if ('traditional' in _l or 'tw' in _l or 'hk' in _l) else 'zh_CN'\n"
         "\n"
         "        return StringBundle(cls.__create_key, locale_str)"),
    ],
}


def locate():
    """返回 {相对路径: 绝对路径}；找不到的项缺席。

    labelImg.py 在 labelImg 包里，其余在顶层的 libs 包里——两个包是分开装的。"""
    found = {}
    try:
        import labelImg
        p = Path(labelImg.__file__).parent / "labelImg.py"
        if p.is_file():
            found["labelImg.py"] = p
    except Exception:
        pass
    try:
        import libs
        base = Path(libs.__file__).parent
        for name in ("canvas.py", "shape.py", "yolo_io.py", "stringBundle.py"):
            p = base / name
            if p.is_file():
                found[f"libs/{name}"] = p
    except Exception:
        pass
    return found


def main():
    found = locate()
    if not found:
        print("❌ 找不到 labelImg。先装：pip install labelImg")
        return 1

    total_fixed = total_already = 0
    for rel, patches in PATCHES.items():
        path = found.get(rel)
        if path is None:
            print(f"⚠️  没找到 {rel}，跳过")
            continue

        text = path.read_text(encoding="utf-8")
        todo = [(o, n) for o, n in patches if o in text and n not in text]
        already = sum(1 for o, n in patches if n in text)

        if not todo:
            print(f"✓  {rel}：已是修好的版本（{already}/{len(patches)} 处）")
            total_already += already
            continue

        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copyfile(path, bak)

        for o, n in todo:
            text = text.replace(o, n)
        path.write_text(text, encoding="utf-8")
        print(f"🔧 {rel}：修了 {len(todo)} 处"
              + (f"（另有 {already} 处本来就是好的）" if already else "")
              + f"  备份 → {bak.name}")
        total_fixed += len(todo)

    print()
    if total_fixed:
        print(f"✅ 共修复 {total_fixed} 处。重新启动标注软件即可：")
        print("   python scripts/open_labelimg.py <批次名>")
    else:
        print(f"✅ 全部 {total_already} 处本来就是好的，不用改。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
