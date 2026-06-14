"""Tkinter GUI：全屏视频 + 半透明悬浮 UI（圆角按钮 / 悬浮清单 / 顶栏）。

显示模型（2026-06 改版）：
  整个窗口是一张全屏视频（铺满裁切）。底部圆角按钮、右侧工具清单、顶部信息条
  都用 PIL 半透明"画"进每一帧里，透出底下画面、不挡视频。Tkinter 原生控件做不到
  真透明（背景是不透明色块），所以走 PIL 合成这条路。
  - 按钮可点：画布坐标命中检测（_on_click / _on_motion）触发命令 + 悬停高亮。
  - 不卡的关键：顶栏/清单/按钮都做成"缓存贴图（RGBA sprite）"，只在内容变化时重绘；
    每帧只把缓存贴图 alpha 混合到画面的小块区域（numpy，便宜），绝不每帧整幅 PIL 渲染。

检测后端 / 双线程实时逻辑与之前一致（显示线程读帧+叠框，推理线程后台跑 YOLO，
主线程 _ui_pump 取帧画图）。详见各方法注释。
"""
import math
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

from .detector import ToolDetector
from .loader import load_tools, load_tool_names
from .utils import DATA_DIR, imread_unicode
from .voice import VoiceReporter

UI_REFRESH_INTERVAL = 3.0   # 列表 UI 刷新节奏
DETECT_INTERVAL = 0.4       # 检测限频：每 N 秒推理一次（其余时间只读帧+叠最近的框，保证视频顺）
VISIBILITY_WINDOW = 5.0     # 工具可见性窗口（去抖）：最近 N 秒内识别到就算"在"。
SPEAK_INTERVAL = 8.0        # 语音播报间隔

# ---- 配色（统一暗色主题）----
C_BG = "#0f1629"            # 窗口主背景（深海军蓝）—— 无画面时的底
C_ACCENT = "#38bdf8"        # 主题强调色（青）
C_TEXT = "#e6edf3"          # 主文字
C_MUTED = "#7d8aa0"         # 次要/未识别文字
C_GREEN = "#22c55e"         # 识别成功
C_AMBER = "#f59e0b"         # 中等置信度
C_RED = "#ef4444"           # 未识别
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"   # 微软雅黑（中文）

# 按钮配色 (常态, 悬停)
BTN_CAM = ("#16a34a", "#22c55e")
BTN_VIDEO = ("#2563eb", "#3b82f6")
BTN_IMAGE = ("#0891b2", "#06b6d4")
BTN_STOP = ("#dc2626", "#ef4444")
BTN_RESET = ("#475569", "#64748b")

# ---- 悬浮层尺寸 ----
TOPBAR_H = 76               # 顶部文字区高度（无背景条，仅占位排版）
TITLE_SPACING = 16          # 标题字间距
TITLE_SIZE = 32             # 顶部标题字号
HINT_SIZE = 22              # 中间提示字号（须小于标题）
PANEL_W = 372              # 右侧悬浮清单宽度
WIN_W, WIN_H = 2184, 1300  # 启动窗口尺寸
STRIP_H = 138              # 底部按钮条高度
PILL_H = 70                # 圆角按钮高度
PILL_PAD = 54             # 按钮文字左右内边距（越大按钮越长）
PILL_GAP = 18             # 按钮间距
MARGIN = 22               # 边距

DIAG_PATH = DATA_DIR / "run_diag.txt"


def diag(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    try:
        with open(DIAG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class ToolDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智能工具检测系统")
        # 启动时在屏幕中央显示
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        px, py = max(0, (sw - WIN_W) // 2), max(0, (sh - WIN_H) // 2)
        self.root.geometry(f"{WIN_W}x{WIN_H}+{px}+{py}")
        self.root.minsize(1100, 680)
        self.root.configure(bg=C_BG)

        self.voice = VoiceReporter()

        # 后端选择：data/yolo/best.pt 存在 → 用 YOLO；否则回退 SIFT 模板匹配。
        yolo_model = DATA_DIR / "yolo" / "best.pt"
        self.use_yolo = yolo_model.is_file()
        self.detector_engine = None
        self.model_error = None
        try:
            if self.use_yolo:
                from .yolo_detector import YoloToolDetector
                class_map = DATA_DIR / "yolo" / "class_map.json"
                self.detector_engine = YoloToolDetector(str(yolo_model), str(class_map))
                self.tools = [{"name": n} for n in load_tool_names()]
                print(f"✅ YOLO 模式：{yolo_model}，{len(self.tools)} 个工具")
            else:
                self.tools = load_tools(verbose=True)
                self.detector_engine = ToolDetector(self.tools)
                print(f"✅ SIFT 模式：{len(self.tools)} 个工具模板")
        except Exception as e:
            self.model_error = f"{'YOLO' if self.use_yolo else 'SIFT'} 加载失败: {e}"
            print(self.model_error)
            try:
                self.use_yolo = False
                self.tools = load_tools(verbose=True)
                self.detector_engine = ToolDetector(self.tools)
                self.model_error = None
                print("↩️  已回退到 SIFT 模式")
            except Exception as e2:
                self.tools = []
                self.model_error = f"检测器加载失败: {e2}"
                print(self.model_error)

        # 运行状态
        self.cap = None
        self.use_camera = False
        self.is_video_file = False
        self.video_fps = 0.0
        self.static_frame = None
        self.running = False
        self.should_speak = False
        self.last_seen = {}
        self.last_counts = {}
        self.last_source = {}
        self.detect_threshold = 10

        # 双线程实时
        self._last_boxes = []
        self._box_store = {}
        self._last_annotated = None
        self._latest_frame = None
        self._display_frame = None
        self._display_id = 0
        self._frame_lock = threading.Lock()
        self._infer_running = False
        self._last_shown_id = -1
        self._last_recog = None
        self._last_ui_refresh = 0.0
        self._last_speak = 0.0

        # ---- 悬浮层状态 ----
        self._cw = self._ch = 0          # 当前显示区像素尺寸
        self._cur_base = None            # 最近一帧底图（BGR）或 None（空闲暗底）
        self._font_cache = {}
        self._backend_label = "YOLO 深度学习" if self.use_yolo else "SIFT 模板匹配"
        self._status_text = "点击下方按钮开始检测"
        self._status_color = C_MUTED
        # 工具视图：[(name, recog, count)]；识别优先排序
        self._tool_view = [(t["name"], False, 0) for t in self.tools]
        self._recog_count = 0
        # 按钮表（顺序即从左排列；group L 左、R 右）
        self._buttons = [
            {"key": "cam",   "text": "摄像头",  "colors": BTN_CAM,   "command": self.detect_camera, "group": "L", "enabled": True},
            {"key": "video", "text": "视频",    "colors": BTN_VIDEO, "command": self.open_video,    "group": "L", "enabled": True},
            {"key": "image", "text": "图片",    "colors": BTN_IMAGE, "command": self.detect_image,  "group": "L", "enabled": True},
            {"key": "stop",  "text": "停止",    "colors": BTN_STOP,  "command": self.stop_detection,"group": "R", "enabled": False},
            {"key": "clear", "text": "清空",    "colors": BTN_RESET, "command": self.reset_display, "group": "R", "enabled": True},
        ]
        for b in self._buttons:
            b["rect"] = None
        self._hover_key = None
        self._show_panel = False         # 右侧清单只在检测时显示（开始检测前/清空后隐藏）
        # 贴图缓存 + 签名
        self._sp_topbar = self._sp_panel = self._sp_buttons = None
        self._sig_topbar = self._sig_panel = self._sig_buttons = None
        self._panel_xy = (0, 0)

        diag("=" * 50)
        diag(f"启动诊断：后端={'YOLO' if self.use_yolo else 'SIFT(回退)'}, "
             f"cpu_count={os.cpu_count()}, 工具数={len(self.tools)}, "
             f"加载错误={self.model_error or '无'}")

        self.create_ui()

    # ---------------- UI 搭建 ----------------

    def create_ui(self):
        # 全屏视频底板：一张铺满窗口的 Label，悬浮 UI 都画进它显示的图里
        self.video_label = tk.Label(self.root, bg=C_BG, bd=0)
        self.video_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.video_label.bind("<Configure>", self._on_resize)
        self.video_label.bind("<Button-1>", self._on_click)
        self.video_label.bind("<Motion>", self._on_motion)
        # 初次绘制（等尺寸 realize 后）
        self.root.after(60, self._recompose_idle)

    def _font(self, size, bold=False):
        key = (size, bold)
        f = self._font_cache.get(key)
        if f is None:
            try:
                # msyh.ttc：0=Regular 1=Bold
                f = ImageFont.truetype(FONT_PATH, size, index=1 if bold else 0)
            except Exception:
                f = ImageFont.load_default()
            self._font_cache[key] = f
        return f

    @staticmethod
    def _text_wh(font, text):
        l, t, r, b = font.getbbox(text)
        return r - l, b - t

    # ---------------- 贴图（sprite）构建 ----------------

    def _make_topbar(self, w):
        """顶部仅展示标题「智能工具检测系统」，水平居中、字间距拉开。
        无背景条——文字直接透明浮在画面上，描边阴影保证亮背景下也看得清。"""
        im = Image.new("RGBA", (w, TOPBAR_H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        f = self._font(TITLE_SIZE, bold=True)
        title = "智能工具检测系统"
        widths = [self._text_wh(f, ch)[0] for ch in title]
        total_w = sum(widths) + TITLE_SPACING * (len(title) - 1)
        th = self._text_wh(f, title)[1]
        x = (w - total_w) // 2
        y = (TOPBAR_H - th) // 2 - 1
        for ch, cw in zip(title, widths):
            # 厚黑描边 + 纯白字，亮/暗背景下都清晰
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-2, -1, 0, 1, 2):
                    if dx or dy:
                        d.text((x + dx, y + dy), ch, font=f, fill=(0, 0, 0, 210))
            d.text((x, y), ch, font=f, fill=(255, 255, 255))
            x += cw + TITLE_SPACING
        return np.array(im)

    def _make_panel(self):
        """右侧悬浮清单：半透明方角面板，识别在上、未识别在下。"""
        rows = self._tool_view
        head_h = 56
        row_h = 48
        ph = head_h + len(rows) * row_h + 16
        im = Image.new("RGBA", (PANEL_W, ph), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        # 更透明、方角（无圆角）
        d.rectangle([0, 0, PANEL_W - 1, ph - 1], fill=(22, 33, 62, 110))
        fh = self._font(18, bold=True)
        d.text((18, 16), "已识别工具", font=fh, fill=_hex2rgb(C_TEXT))
        # 右侧：进度 / 总数（如 6/9）
        total = len(self._valid_tools())
        prog = f"{self._recog_count}/{total}"
        fbd = self._font(18, bold=True)
        pw = self._text_wh(fbd, prog)[0]
        d.text((PANEL_W - 18 - pw, 16), prog, font=fbd,
               fill=_hex2rgb(C_GREEN if self._recog_count else C_MUTED))
        fn = self._font(16, bold=True)
        fc = self._font(14, bold=True)
        y = head_h
        for name, recog, count in rows:
            # 状态点：识别=绿实心圆，未识别=红空心圈（画图形，不依赖字体字形，避免豆腐块）
            cy = y + row_h // 2
            cx, r = 26, 6
            if recog:
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_hex2rgb(C_GREEN))
            else:
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=_hex2rgb(C_RED), width=2)
            d.text((48, y + row_h // 2 - 12), name, font=fn,
                   fill=_hex2rgb(C_TEXT if recog else C_MUTED))
            if self.use_yolo and recog and count > 0:
                cc = C_GREEN if count >= 70 else C_AMBER if count >= 50 else C_MUTED
                txt = f"{count}%"
            else:
                cc = "#4a5670"
                txt = "--"
            tw = self._text_wh(fc, txt)[0]
            d.text((PANEL_W - 18 - tw, y + row_h // 2 - 10), txt, font=fc, fill=_hex2rgb(cc))
            y += row_h
        return np.array(im)

    def _make_buttons(self, w, h):
        """底部圆角半透明按钮条：五个按钮整体居中排成一排。
        同时把每个按钮的绝对矩形写回 b['rect'] 供点击命中。"""
        im = Image.new("RGBA", (w, STRIP_H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        f = self._font(22, bold=True)
        ly = (STRIP_H - PILL_H) // 2
        abs_y = h - STRIP_H + ly

        # 先量出每个按钮宽度，算总宽以便居中
        widths = [self._text_wh(f, b["text"])[0] + 2 * PILL_PAD for b in self._buttons]
        total = sum(widths) + PILL_GAP * (len(self._buttons) - 1)
        x = (w - total) // 2

        for b, bw in zip(self._buttons, widths):
            base, hover = b["colors"]
            if not b["enabled"]:
                rgb, a, tcol = _hex2rgb(base), 90, (154, 166, 189)
            elif self._hover_key == b["key"]:
                rgb, a, tcol = _hex2rgb(hover), 235, (255, 255, 255)
            else:
                rgb, a, tcol = _hex2rgb(base), 205, (255, 255, 255)
            d.rounded_rectangle([x, ly, x + bw, ly + PILL_H], radius=PILL_H // 2,
                                fill=(*rgb, a))
            th = self._text_wh(f, b["text"])[1]
            d.text((x + PILL_PAD, ly + (PILL_H - th) // 2 - 2), b["text"], font=f, fill=tcol)
            b["rect"] = (x, abs_y, bw, PILL_H)
            x += bw + PILL_GAP
        return np.array(im)

    def _ensure_sprites(self):
        """按签名惰性重建三块贴图；尺寸/状态没变就复用缓存。"""
        w, h = self._cw, self._ch
        sig_t = (w,)   # 顶部只剩居中标题，只随宽度变化
        if sig_t != self._sig_topbar:
            self._sp_topbar = self._make_topbar(w)
            self._sig_topbar = sig_t

        sig_p = (tuple(self._tool_view), self._recog_count, self.use_yolo)
        if sig_p != self._sig_panel:
            self._sp_panel = self._make_panel()
            self._sig_panel = sig_p
        self._panel_xy = (w - PANEL_W - MARGIN, MARGIN)   # 贴右上角

        sig_b = (w, h, self._hover_key, tuple(b["enabled"] for b in self._buttons))
        if sig_b != self._sig_buttons:
            self._sp_buttons = self._make_buttons(w, h)
            self._sig_buttons = sig_b

    # ---------------- 合成 + 显示 ----------------

    @staticmethod
    def _cover_resize(frame_bgr, w, h):
        """铺满裁切：等比放大到盖住 w×h，居中裁掉溢出。返回 BGR(w×h)。"""
        fh, fw = frame_bgr.shape[:2]
        scale = max(w / fw, h / fh)
        nw, nh = max(w, int(math.ceil(fw * scale))), max(h, int(math.ceil(fh * scale)))
        resized = cv2.resize(frame_bgr, (nw, nh))
        x0, y0 = (nw - w) // 2, (nh - h) // 2
        return resized[y0:y0 + h, x0:x0 + w]

    @staticmethod
    def _blend(dst_rgb, sp_rgba, x, y):
        """把 RGBA 贴图 alpha 混合到 dst_rgb 的 (x,y)。仅作用于重叠子块，便宜。"""
        H, W = sp_rgba.shape[:2]
        dh, dw = dst_rgb.shape[:2]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + W, dw), min(y + H, dh)
        if x1 <= x0 or y1 <= y0:
            return
        sp = sp_rgba[y0 - y:y1 - y, x0 - x:x1 - x]
        a = sp[..., 3:4].astype(np.float32) / 255.0
        reg = dst_rgb[y0:y1, x0:x1].astype(np.float32)
        dst_rgb[y0:y1, x0:x1] = (reg * (1 - a) + sp[..., :3].astype(np.float32) * a).astype(np.uint8)

    def _compose(self, base_bgr):
        """组装当前显示图 = 底图(视频帧或暗底) + 顶栏/清单/按钮，画到 video_label。"""
        w, h = self._cw, self._ch
        if w < 40 or h < 40:
            return
        if base_bgr is None:
            rgb = np.empty((h, w, 3), np.uint8)
            rgb[:] = _hex2rgb(C_BG)
        else:
            rgb = cv2.cvtColor(self._cover_resize(base_bgr, w, h), cv2.COLOR_BGR2RGB)

        self._ensure_sprites()
        self._blend(rgb, self._sp_topbar, 0, 0)
        if self._show_panel and self._sp_panel is not None:
            self._blend(rgb, self._sp_panel, self._panel_xy[0], self._panel_xy[1])
        self._blend(rgb, self._sp_buttons, 0, h - STRIP_H)

        img = Image.fromarray(rgb)
        if base_bgr is None:
            # 空闲：中间提示
            d = ImageDraw.Draw(img)
            hint = "点击下方  摄像头 / 视频 / 图片  开始检测"
            f = self._font(HINT_SIZE)
            tw, th = self._text_wh(f, hint)
            d.text(((w - tw) // 2, (h - th) // 2), hint, font=f, fill=_hex2rgb(C_MUTED))

        imgtk = ImageTk.PhotoImage(img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk, text="")

    def _recompose_idle(self):
        """非运行态（空闲/停止/静态图）重绘一次。运行态由 _ui_pump 逐帧画，不用调这。"""
        self._cw = self.video_label.winfo_width()
        self._ch = self.video_label.winfo_height()
        self._compose(self._cur_base)

    def _show_frame(self, frame_bgr):
        """运行态每帧调用：记下底图并合成显示。"""
        self._cur_base = frame_bgr
        self._cw = self.video_label.winfo_width()
        self._ch = self.video_label.winfo_height()
        self._compose(frame_bgr)

    # ---------------- 交互（resize / 点击 / 悬停）----------------

    def _on_resize(self, event):
        if (event.width, event.height) != (self._cw, self._ch):
            self._cw, self._ch = event.width, event.height
            if not self.running:
                self._recompose_idle()

    def _on_click(self, event):
        for b in self._buttons:
            if not b["enabled"] or not b["rect"]:
                continue
            x, y, bw, bh = b["rect"]
            if x <= event.x <= x + bw and y <= event.y <= y + bh:
                b["command"]()
                return

    def _on_motion(self, event):
        hit = None
        for b in self._buttons:
            if not b["enabled"] or not b["rect"]:
                continue
            x, y, bw, bh = b["rect"]
            if x <= event.x <= x + bw and y <= event.y <= y + bh:
                hit = b["key"]
                break
        if hit != self._hover_key:
            self._hover_key = hit
            self.video_label.configure(cursor="hand2" if hit else "")
            if not self.running:
                self._recompose_idle()

    def set_status(self, text, color=C_MUTED):
        self._status_text = text
        self._status_color = color
        if not self.running:
            self._recompose_idle()

    def _set_buttons_running(self, running):
        for b in self._buttons:
            if b["key"] in ("cam", "video", "image"):
                b["enabled"] = not running
            elif b["key"] == "stop":
                b["enabled"] = running
            # clear 始终可用
        if not running:
            self._hover_key = None
        if not self.running:
            self._recompose_idle()

    def refresh_visible_list(self):
        """重算工具视图（识别优先排序）→ 更新贴图数据。返回识别数。"""
        try:
            now = time.time()
            recognized, unrecognized = [], []
            for tool in self.tools:
                name = tool["name"]
                if now - self.last_seen.get(name, 0.0) < VISIBILITY_WINDOW:
                    recognized.append(name)
                else:
                    unrecognized.append(name)

            view = []
            for name in recognized:
                view.append((name, True, int(self.last_counts.get(name, 0))))
            for name in unrecognized:
                view.append((name, False, int(self.last_counts.get(name, 0))))
            self._tool_view = view
            self._recog_count = len(recognized)

            if not self.running:
                self._recompose_idle()
            return len(recognized)
        except Exception as e:
            print(f"刷新列表出错: {e}")
            return 0

    # ---------------- 启停控制 ----------------

    def _valid_tools(self):
        if self.use_yolo:
            return list(self.tools)
        return [t for t in self.tools if t.get("des") is not None]

    _CAM_ATTEMPTS = [
        (cv2.CAP_DSHOW, "DSHOW", True,  False),
        (cv2.CAP_DSHOW, "DSHOW", True,  True),
        (cv2.CAP_MSMF,  "MSMF",  True,  False),
        (cv2.CAP_MSMF,  "MSMF",  True,  True),
        (cv2.CAP_MSMF,  "MSMF",  False, False),
        (cv2.CAP_DSHOW, "DSHOW", False, False),
    ]

    def _try_open(self, index, flag, force_720p, use_mjpg, warmup_frames):
        cap = cv2.VideoCapture(index, flag)
        if not cap.isOpened():
            cap.release()
            return None, False
        if use_mjpg:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
        if force_720p:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        for _ in range(warmup_frames):
            ret, frame = cap.read()
            if ret and frame is not None and float(frame.std()) > 6.0:
                return cap, True
            time.sleep(0.05)
        cap.release()
        return None, True

    def _open_camera(self, max_index=2, warmup_frames=8):
        for index in range(max_index + 1):
            opened_any = False
            for flag, bname, force_720p, use_mjpg in self._CAM_ATTEMPTS:
                cap, opened = self._try_open(index, flag, force_720p, use_mjpg, warmup_frames)
                opened_any = opened_any or opened
                res = f"{'720p' if force_720p else 'native'}/{'MJPG' if use_mjpg else '默认'}"
                if cap is not None:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    diag(f"摄像头探测：设备{index} {bname} {res} → ✅ 打开成功 {w}x{h}")
                    return cap
                why = "连打开都失败" if not opened else "能打开但读不到真画面(忙/格式不出流)"
                diag(f"摄像头探测：设备{index} {bname} {res} → ✗ {why}")
            if index == 0 and not opened_any:
                diag("摄像头探测：0号所有后端都打不开 → 判定无摄像头子系统，停止探测")
                break
        diag("⚠️ 未找到可用摄像头，回退到选图片模式")
        return None

    def _check_ready(self):
        if self.detector_engine is None:
            messagebox.showwarning(
                "检测器未就绪",
                f"检测器未就绪，无法开始检测。\n\n详细错误：\n{self.model_error or '未知'}")
            return False
        if not self._valid_tools():
            messagebox.showwarning(
                "警告",
                "没有任何可用工具。YOLO 模式请检查 data/yolo/best.pt 和 class_map.json；"
                "SIFT 模式请检查 data/smart_tools.json 和 data/smart_templates/。")
            return False
        return True

    def _reset_frame_state(self):
        self._last_boxes = []
        self._last_annotated = None
        with self._frame_lock:
            self._latest_frame = None
            self._display_frame = None

    def _begin_realtime(self, status_text):
        self.running = True
        self.should_speak = True
        self._show_panel = True          # 检测开始 → 显示右侧清单
        self._reset_frame_state()
        self._set_buttons_running(True)
        self.set_status(status_text, C_GREEN)
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        self._start_ui_pump()

    def detect_camera(self):
        if self.running:
            self.stop_detection()
        if not self._check_ready():
            return
        cap = self._open_camera()
        if cap is None:
            messagebox.showwarning(
                "未找到摄像头",
                "没有检测到可用摄像头。\n请检查摄像头是否插好/被占用，"
                "或改用「视频检测」「图片检测」。")
            return
        self.cap = cap
        self.use_camera = True
        self.is_video_file = False
        self.static_frame = None
        self._begin_realtime("正在检测（摄像头实时）…")

    def detect_image(self):
        if self.running:
            self.stop_detection()
        if not self._check_ready():
            return
        initial_dir = str(DATA_DIR / "samples")
        if not os.path.isdir(initial_dir):
            initial_dir = str(DATA_DIR)
        file_path = filedialog.askopenfilename(
            title="选择要检测的图片",
            initialdir=initial_dir,
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")])
        if not file_path:
            return
        try:
            self.static_frame = imread_unicode(file_path)
        except Exception:
            self.static_frame = None
        if self.static_frame is None:
            messagebox.showerror("错误", f"无法读取图片：{file_path}")
            return

        self.use_camera = False
        self.is_video_file = False
        self.running = True
        self.should_speak = True
        self._show_panel = True          # 图片检测 → 显示右侧清单（结果）
        self._reset_frame_state()
        self._set_buttons_running(True)
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()

    def open_video(self):
        if self.running:
            self.stop_detection()
        if not self._check_ready():
            return
        initial_dir = str(DATA_DIR / "video")
        if not os.path.isdir(initial_dir):
            initial_dir = str(DATA_DIR)
        file_path = filedialog.askopenfilename(
            title="选择要检测的视频",
            initialdir=initial_dir,
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                       ("所有文件", "*.*")],
        )
        if not file_path:
            return
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            messagebox.showerror("错误", f"无法打开视频：\n{file_path}")
            return
        self.cap = cap
        self.use_camera = True
        self.is_video_file = True
        self.static_frame = None
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_fps = fps if fps and fps > 1 else 25.0
        self._begin_realtime(f"正在检测视频（{os.path.basename(file_path)}）…")

    def stop_detection(self):
        self.running = False
        self.should_speak = False
        self._infer_running = False
        self._show_panel = False         # 停止检测 → 隐藏右侧清单
        self.voice.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._set_buttons_running(False)
        self.refresh_visible_list()
        self.set_status("已停止", "#ff6b6b")

    def reset_display(self):
        if self.running:
            self.stop_detection()
        self.last_seen = {}
        self.last_counts = {}
        self._last_boxes = []
        self._last_annotated = None
        self._show_panel = False    # 清空 → 隐藏右侧清单
        self._cur_base = None       # 回到空闲暗底
        try:
            self.refresh_visible_list()
            self.set_status("已清空，待检测", C_MUTED)
        except Exception as e:
            print(f"清空显示出错: {e}")

    # ---------------- 检测循环 ----------------

    def detection_loop(self):
        try:
            if self.use_camera:
                self._camera_loop()
            else:
                self._static_once()
        except Exception as e:
            print(f"检测出错: {e}")
            self.stop_detection()

    def _start_ui_pump(self):
        self._last_shown_id = -1
        self._last_recog = None
        self._last_ui_refresh = 0.0
        self._last_speak = time.time()
        with self._frame_lock:
            self._display_frame = None
            self._display_id = 0
        self.root.after(0, self._ui_pump)

    def _camera_loop(self):
        self._box_store = {}
        self._infer_running = True
        worker = threading.Thread(target=self._inference_worker, daemon=True)
        worker.start()

        frame_interval = (1.0 / self.video_fps) if self.is_video_file else 0.0
        next_frame_t = time.time()
        diag_t0 = time.time()
        diag_frames = 0
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_file:
                    break
                time.sleep(0.05)
                continue
            now = time.time()
            diag_frames += 1
            with self._frame_lock:
                self._latest_frame = frame
            if self.use_yolo:
                display = (self.detector_engine.render_live(frame.copy(), self._last_boxes)
                           if self._last_boxes else frame)
            else:
                display = self._last_annotated if self._last_annotated is not None else frame
            with self._frame_lock:
                self._display_frame = display
                self._display_id += 1
            if now - diag_t0 >= 2.0:
                fps = diag_frames / (now - diag_t0)
                hit = len([n for n, t in list(self.last_seen.items())
                           if now - t < VISIBILITY_WINDOW])
                diag(f"读帧线程：{fps:.1f} fps，窗口内命中 {hit} 个")
                diag_t0 = now
                diag_frames = 0
            if self.is_video_file:
                next_frame_t += frame_interval
                delay = next_frame_t - time.time()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_frame_t = time.time()
            else:
                time.sleep(0.01)

        self._infer_running = False
        if self.is_video_file and self.running:
            self.root.after(0, self._on_video_finished)

    def _ui_pump(self):
        if not self.running:
            return
        now = time.time()
        with self._frame_lock:
            frame = self._display_frame
            did = self._display_id
        if frame is not None and did != self._last_shown_id:
            self._show_frame(frame)
            self._last_shown_id = did

        recog_now = frozenset(n for n, t in list(self.last_seen.items())
                              if now - t < VISIBILITY_WINDOW)
        if ((recog_now != self._last_recog and now - self._last_ui_refresh >= 1.0)
                or now - self._last_ui_refresh >= UI_REFRESH_INTERVAL):
            self.refresh_visible_list()
            self._last_ui_refresh = now
            self._last_recog = recog_now

        if self.should_speak and now - self._last_speak >= SPEAK_INTERVAL:
            self.speak_missing()
            self._last_speak = now

        self.root.after(15, self._ui_pump)

    def _on_video_finished(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._set_buttons_running(False)
        self.refresh_visible_list()
        self.set_status("视频检测完毕", "#00ff88")

    def _hold_boxes(self, boxes):
        now = time.time()
        for b in boxes:
            self._box_store[b[4]] = (b, now)
        held = []
        for name in list(self._box_store.keys()):
            b, ts = self._box_store[name]
            if now - ts < VISIBILITY_WINDOW:
                held.append(b)
            else:
                del self._box_store[name]
        return held

    def _inference_worker(self):
        last_detect = 0.0
        diag_t0 = time.time()
        diag_infer_ms = 0.0
        diag_infers = 0
        while self._infer_running and self.running:
            now = time.time()
            if now - last_detect < DETECT_INTERVAL:
                time.sleep(0.01)
                continue
            with self._frame_lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
            if frame is None:
                time.sleep(0.01)
                continue
            last_detect = now
            t_infer = time.time()
            try:
                if self.use_yolo:
                    detected, counts, boxes = self.detector_engine.infer(frame)
                    self._last_boxes = self._hold_boxes(boxes)
                else:
                    detected, annotated, counts = self.detector_engine.detect(frame)
                    self._last_annotated = annotated
            except Exception as e:
                diag(f"检测出错: {e}")
                detected, counts = set(), {}
            diag_infer_ms += (time.time() - t_infer) * 1000
            diag_infers += 1
            stamp = time.time()
            for name in detected:
                self.last_seen[name] = stamp
            self.last_counts.update(counts)
            self.last_source = dict(getattr(self.detector_engine, "last_source", {}))
            self.detect_threshold = getattr(self.detector_engine, "last_threshold", 0)
            hits = " ".join(
                f"{n}={counts.get(n, 0)}{self.last_source.get(n, '')}"
                for n in sorted(detected))
            print(f"[检测 {len(detected)}个] {hits or '(空)'}")
            if stamp - diag_t0 >= 2.0:
                avg = diag_infer_ms / max(diag_infers, 1)
                diag(f"推理线程：{avg:.0f} ms/帧 × {diag_infers} 次/2s")
                diag_t0 = stamp
                diag_infer_ms = 0.0
                diag_infers = 0

    def _static_once(self):
        now = time.time()
        detected, display, counts = self.detector_engine.detect(self.static_frame)
        for name in detected:
            self.last_seen[name] = now
        self.last_counts.update(counts)
        self.last_source = dict(getattr(self.detector_engine, "last_source", {}))
        self.detect_threshold = getattr(self.detector_engine, "last_threshold", 0)
        recognized_count = self.refresh_visible_list()
        if self.should_speak:
            self.speak_missing()
        self.running = False
        try:
            self._set_buttons_running(False)
            total = len(self._valid_tools())
            self._status_text = f"检测完成：识别 {recognized_count}/{total}"
            self._status_color = "#00ff88"
            self._cur_base = display          # 静态图作底图
            self._recompose_idle()            # 末尾统一重绘（此时 running=False）
        except Exception:
            pass

    # ---------------- 语音 ----------------

    def speak_missing(self):
        all_tools = set(t["name"] for t in self._valid_tools())
        now = time.time()
        currently_visible = {
            name for name, t in self.last_seen.items()
            if now - t < VISIBILITY_WINDOW
        }
        missing = all_tools - currently_visible
        self.voice.report(missing)


def main():
    root = tk.Tk()
    ToolDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
