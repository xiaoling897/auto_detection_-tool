"""Tkinter GUI：实时扫描模式（检测后端自动选择：有 data/yolo/best.pt 用 YOLO，否则 SIFT；3 秒滑动窗口防抖）

工作流：
  - 摄像头模式（双线程实时）：显示线程持续读帧 + 叠最近的框 + 显示，永远满帧率；
    推理线程在后台跑 YOLO，只回写"最近的框 + 识别时间戳"。两线程不互相阻塞，
    所以模型再重也只拖慢"框刷新率"，视频本身始终流畅。
    避开旧双线程"饿死显示"的两个关键：① torch 限核（留 CPU 给显示线程，见
    yolo_detector）；② 显示线程每帧只做 cv2 画框 + 缓存标签贴图（render_live），
    绝不每帧整幅 PIL 渲染。
  - 静态图模式：弹文件框选图 → 检测一次 → 立即刷新列表
  - 右侧列表全量显示全部工具，识别中的（✓ 绿）排上面，未识别的（✗ 红）排下面
  - 列表 UI 每 3 秒刷新一次（不实时刷新，避免闪烁）；可见性窗口 3 秒
  - 检测出结果后等待 3 秒，再播报当前工具齐全/缺失状态
  - "🔄 清空显示"先停检测再清空 last_seen

依赖 data/smart_tools.json 里的工具配置 + data/smart_templates/ 下的模板图。
"""
import json
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from .detector import ToolDetector
from .loader import load_tools, load_tool_names
from .utils import DATA_DIR, imread_unicode
from .voice import VoiceReporter

UI_REFRESH_INTERVAL = 3.0   # 列表 UI 刷新节奏
DETECT_INTERVAL = 0.4       # 检测限频：每 N 秒推理一次（其余时间只读帧+叠最近的框，保证视频顺）
VISIBILITY_WINDOW = 3.0     # 工具可见性窗口（去抖）：最近 N 秒内识别到就算"在"。
                            # 设大一点能吸收"检测抖动 + 伸手遮挡"导致的瞬时漏检，
                            # 避免拿走一个工具时把旁边被手挡住的工具误报成缺失。
                            # 太小→误报缺失；太大→真拿走后要等更久才报。同时影响列表/框/语音。
SPEAK_INTERVAL = 3.0        # 语音播报间隔
BAD_FRAME_LIMIT = 3         # 「读帧失败」连续这么多次 → 判定信号丢失（拔线/断流，立刻要报）
BLANK_FRAME_LIMIT = 25      # 「读到但全黑」要连续这么多次才判定信号丢失（≈1s 以上）。
                            # 必须比 BAD_FRAME_LIMIT 大很多：摄像头刚打开时自动曝光没起来，
                            # 头几十帧就是黑的，用 3 帧判定会在开机瞬间闪一下"无信号"橙屏。
CAMERA_WARMUP_GRACE = 3.0   # 摄像头开流后的宽限期：这段时间内的黑帧一律不算"信号丢失"，
                            # 专治"刚打开闪一下橙屏再恢复"的闪屏。
BLANK_MEAN_THRESHOLD = 6.0  # 接近全黑的平均亮度阈值
BLANK_STD_THRESHOLD = 4.0   # 接近纯色黑屏的纹理/方差阈值

# 视频显示画布固定尺寸（16:9）。固定住有两个作用：
#   ① 每帧 letterbox 到同一尺寸 → PhotoImage 只建一次、后续 paste 原地更新，不再每帧
#      configure(image=...)（那会让 Tk 擦背景+重算布局，就是"画面一闪一闪"的主因）；
#   ② 画面尺寸恒定 → 左侧视频区不会随分辨率/文字占位符变大变小把右侧清单挤得来回跳。
#   尺寸取 900x506：是在最小窗口(minsize 1200x760，视频区实测可用 900x588)下**不被裁切的
#   最大 16:9 画布**，同时和 720p 摄像头同比例、缩放后正好铺满不留黑边。要改先量可用区。
DISPLAY_W, DISPLAY_H = 900, 506
CANVAS_BG_BGR = (26, 14, 10)   # = C_VIDEO #0a0e1a 的 BGR，留黑边和视频区同色

CAMERA_PROBE_BUDGET = 10.0  # 整轮探测的总时间预算（秒）。最坏情况（几个设备号都能打开
                            # 却都不出流）也不会无限拖，超时就停下给明确提示。

# 上次成功打开摄像头的配置缓存：命中后下次直接一步打开（省掉整轮探测的十几秒）。
CAMERA_PREF_PATH = DATA_DIR / "camera_pref.json"

# ---- 配色（统一暗色主题，集中放这里方便整体换肤）----
C_BG = "#0f1629"            # 窗口主背景（深海军蓝）
C_PANEL = "#16213e"         # 面板背景
C_CARD = "#1d2b4a"          # 卡片/工具行背景（比面板亮一点）
C_VIDEO = "#0a0e1a"         # 视频区背景
C_BORDER = "#2a3a5c"        # 描边
C_ACCENT = "#38bdf8"        # 主题强调色（青）
C_TEXT = "#e6edf3"          # 主文字
C_MUTED = "#7d8aa0"         # 次要/未识别文字
C_GREEN = "#22c55e"         # 识别成功
C_AMBER = "#f59e0b"         # 中等置信度
FONT = "Microsoft YaHei UI"

# 按钮配色 (常态, 悬停)
BTN_CAM = ("#16a34a", "#22c55e")
BTN_VIDEO = ("#2563eb", "#3b82f6")
BTN_IMAGE = ("#0891b2", "#06b6d4")
BTN_STOP = ("#dc2626", "#ef4444")
BTN_RESET = ("#475569", "#64748b")

# 诊断日志：写到 exe 同级 data/run_diag.txt（windowed 包没控制台也能留痕），同时打印。
# 排查"卡 + 识别不出"用——记录后端、单次推理耗时、检测速率、显示帧率、读帧耗时。
DIAG_PATH = DATA_DIR / "run_diag.txt"


def diag(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        print(line)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = line.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe)
    try:
        with open(DIAG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


class ToolDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智能工具检测系统")
        self.root.geometry("1480x920")
        self.root.minsize(1200, 760)
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
                # YOLO 模式下工具清单来自 class_map（中文名），不需要模板特征
                self.tools = [{"name": n} for n in load_tool_names()]
                print(f"✅ YOLO 模式：{yolo_model}，{len(self.tools)} 个工具")
            else:
                self.tools = load_tools(verbose=True)
                self.detector_engine = ToolDetector(self.tools)
                print(f"✅ SIFT 模式：{len(self.tools)} 个工具模板")
        except Exception as e:
            # YOLO 失败时回退 SIFT，让程序仍能跑
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
        self.is_video_file = False   # True=播放视频文件（需按原生帧率播放、放完自动停）
        self.video_fps = 0.0
        self.static_frame = None
        self.running = False
        self.should_speak = False
        self.opening_camera = False
        self._camera_open_token = 0
        self.last_seen = {}      # 工具名 -> 上次识别到的时间戳
        self.last_counts = {}    # 工具名 -> 上一帧 RANSAC 内点数
        self.last_source = {}    # 工具名 -> 上一帧命中来源（"强"/"弱"/"色"）
        self.detect_threshold = 10   # 上一帧有效阈值（自适应，诊断显示用）

        # 双线程实时：显示线程(读帧+叠框+显示) 与 推理线程(后台跑 YOLO) 解耦。
        # 重模型(如 yolo11s)单帧推理几十~几百 ms，若和显示同线程会每隔 DETECT_INTERVAL
        # 就把画面冻住几百 ms → 卡顿。拆开后推理只在后台更新"最近的框"，显示线程不等推理，
        # 视频始终满帧率；模型重只表现为框刷新率低一点（固定监控场景工具不动，无感）。
        self._last_boxes = []            # 显示线程每帧叠到实时画面的框（经"存活期"平滑，防闪）
        self._box_store = {}             # 工具名 -> (框, 时间戳)：框防闪用，见 _hold_boxes
        self._last_annotated = None      # 最近一次 SIFT 标注帧（SIFT 回退用）
        self._latest_frame = None        # 读帧线程读到的最新原始帧，供推理线程取用（共享）
        self._display_frame = None       # 叠好框的"待显示帧"，由主线程 _ui_pump 取去画（共享）
        self._display_id = 0             # 待显示帧的版本号，主线程据此判断是否有新帧
        self._frame_lock = threading.Lock()
        self._infer_running = False      # 推理线程开关
        # 主线程 UI 泵的状态
        self._last_shown_id = -1
        self._last_recog = None
        self._last_ui_refresh = 0.0
        self._last_speak = None
        self._has_detection_sample = False
        self._camera_signal_lost = False
        self._imgtk = None       # 视频区复用的 PhotoImage（见 _show_frame）

        diag("=" * 50)
        diag(f"启动诊断：后端={'YOLO' if self.use_yolo else 'SIFT(回退)'}, "
             f"cpu_count={os.cpu_count()}, 工具数={len(self.tools)}, "
             f"加载错误={self.model_error or '无'}")

        self.create_ui()

    # ---------------- UI ----------------

    def _make_button(self, parent, text, colors, command, state="normal"):
        """扁平现代风按钮 + 悬停高亮。colors=(常态色, 悬停色)。"""
        base, hover = colors
        btn = tk.Button(parent, text=text, command=command, state=state,
                        font=(FONT, 13, "bold"), bg=base, fg="white",
                        activebackground=hover, activeforeground="white",
                        disabledforeground="#9aa6bd",
                        relief="flat", bd=0, cursor="hand2", padx=16, pady=11)
        btn._base, btn._hover = base, hover
        btn.bind("<Enter>",
                 lambda e: btn.config(bg=hover) if str(btn["state"]) == "normal" else None)
        btn.bind("<Leave>",
                 lambda e: btn.config(bg=base) if str(btn["state"]) == "normal" else None)
        return btn

    def _set_buttons_running(self, running):
        """运行中：禁用三个检测按钮、启用停止；停止时反之。顺带复位悬停残留色。"""
        for b in (self.cam_btn, self.video_btn, self.image_btn):
            b.config(state="disabled" if running else "normal", bg=b._base)
        self.stop_btn.config(state="normal" if running else "disabled",
                             bg=self.stop_btn._base)

    def create_ui(self):
        main = tk.Frame(self.root, bg=C_BG)
        main.pack(fill="both", expand=True)

        # ---- 顶部标题栏 ----
        header = tk.Frame(main, bg=C_BG)
        header.pack(fill="x", padx=22, pady=(16, 6))
        tk.Label(header, text="🔧 智能工具检测系统", font=(FONT, 24, "bold"),
                 bg=C_BG, fg=C_ACCENT).pack(side="left")
        badge_bg = "#14532d" if self.use_yolo else "#5b3a13"
        badge_fg = C_GREEN if self.use_yolo else C_AMBER
        self.backend_badge = tk.Label(
            header, text=f"  {'YOLO 深度学习' if self.use_yolo else 'SIFT 模板匹配'}  ",
            font=(FONT, 10, "bold"), bg=badge_bg, fg=badge_fg, padx=4, pady=3)
        self.backend_badge.pack(side="right", pady=6)

        # ---- 主体：左视频 + 右列表 ----
        body = tk.Frame(main, bg=C_BG)
        body.pack(fill="both", expand=True, padx=22, pady=(0, 16))

        left = tk.Frame(body, bg=C_BG)
        left.pack(side="left", fill="both", expand=True)

        # 视频卡片（细描边）
        video_card = tk.Frame(left, bg=C_VIDEO, highlightbackground=C_BORDER,
                              highlightthickness=1, bd=0)
        video_card.pack(fill="both", expand=True)
        # 关掉尺寸传播：不让里面的图片/文字反过来撑大卡片。
        # 否则"文字占位符(很小) → 第一帧画面(900x506)"这一下会把左栏撑开、右侧清单被挤，
        # 整个界面抖一下 —— 这也是"打开摄像头时闪一下"的一部分。
        video_card.pack_propagate(False)
        video_card.configure(width=DISPLAY_W + 14, height=DISPLAY_H + 14)
        self.video_label = tk.Label(video_card, bg=C_VIDEO, fg=C_MUTED,
                                    text="📷  点击下方按钮开始检测",
                                    font=(FONT, 15))
        self.video_label.pack(fill="both", expand=True, padx=6, pady=6)

        # ---- 按钮栏 ----
        bar = tk.Frame(left, bg=C_BG)
        bar.pack(fill="x", pady=(14, 0))
        self.cam_btn = self._make_button(bar, "📷 摄像头检测", BTN_CAM, self.detect_camera)
        self.cam_btn.pack(side="left")
        self.video_btn = self._make_button(bar, "🎬 视频检测", BTN_VIDEO, self.open_video)
        self.video_btn.pack(side="left", padx=10)
        self.image_btn = self._make_button(bar, "🖼 图片检测", BTN_IMAGE, self.detect_image)
        self.image_btn.pack(side="left")
        # 弹簧：把"停止/清空"推到右边
        tk.Frame(bar, bg=C_BG).pack(side="left", expand=True, fill="x")
        self.stop_btn = self._make_button(bar, "⏹ 停止检测", BTN_STOP,
                                          self.stop_detection, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 10))
        self.reset_btn = self._make_button(bar, "🗑 清空显示", BTN_RESET, self.reset_display)
        self.reset_btn.pack(side="left")

        # ---- 右侧工具清单 ----
        right = tk.Frame(body, bg=C_PANEL, width=360)
        right.pack(side="right", fill="y", padx=(18, 0))
        right.pack_propagate(False)

        rhead = tk.Frame(right, bg=C_PANEL)
        rhead.pack(fill="x", padx=18, pady=(18, 8))
        tk.Label(rhead, text="📋 已识别工具", font=(FONT, 15, "bold"),
                 bg=C_PANEL, fg=C_TEXT).pack(side="left")
        self.count_badge = tk.Label(rhead, text="0/0", font=(FONT, 12, "bold"),
                                    bg=C_CARD, fg=C_GREEN, padx=10, pady=2)
        self.count_badge.pack(side="right")

        self.tool_status_frame = tk.Frame(right, bg=C_PANEL)
        self.tool_status_frame.pack(fill="both", expand=True, padx=12, pady=4)

        self.build_tool_rows()

        self.stats_label = tk.Label(right, text="待检测…", font=(FONT, 11),
                                    bg=C_PANEL, fg=C_MUTED)
        self.stats_label.pack(pady=(6, 16))

        # 初始状态：所有工具显示为未识别——必须放在 stats_label 创建之后
        self.refresh_visible_list()

    def build_tool_rows(self):
        """为每个工具预创建一张"卡片行"。refresh_visible_list 决定 pack 顺序 + 状态颜色。"""
        for tool in self.tools:
            name = tool["name"]
            row = tk.Frame(self.tool_status_frame, bg=C_CARD)

            # 左侧状态色条
            accent = tk.Frame(row, bg="#ef4444", width=4)
            accent.pack(side="left", fill="y")

            inner = tk.Frame(row, bg=C_CARD)
            inner.pack(side="left", fill="both", expand=True, padx=(10, 10), pady=9)

            status_indicator = tk.Label(inner, text="✗", font=("Segoe UI Emoji", 13),
                                        bg=C_CARD, fg="#ef4444", width=2)
            status_indicator.pack(side="left")

            name_label = tk.Label(inner, text=name, font=(FONT, 11, "bold"),
                                  bg=C_CARD, fg=C_MUTED, anchor="w")
            name_label.pack(side="left", fill="x", expand=True, padx=(6, 0))

            match_label = tk.Label(inner, text="--", font=(FONT, 10, "bold"),
                                   bg=C_CARD, fg=C_MUTED, width=8, anchor="e")
            match_label.pack(side="right")

            tool["row"] = row
            tool["accent"] = accent
            tool["name_label"] = name_label
            tool["status_label"] = status_indicator
            tool["match_label"] = match_label

    def refresh_visible_list(self):
        """全部工具都显示：识别到的（✓ 绿）排上面，未识别的（✗ 红）排下面。

        识别判定：last_seen 在最近 VISIBILITY_WINDOW 秒内。
        每次刷新都 forget 所有行再按"识别优先"顺序重新 pack——保证排序稳定。
        """
        try:
            now = time.time()
            for tool in self.tools:
                row = tool.get("row")
                if row is not None:
                    row.pack_forget()

            recognized = []
            unrecognized = []
            for tool in self.tools:
                if tool.get("row") is None:
                    continue
                name = tool["name"]
                if now - self.last_seen.get(name, 0.0) < VISIBILITY_WINDOW:
                    recognized.append(tool)
                else:
                    unrecognized.append(tool)

            for tool in recognized + unrecognized:
                row = tool["row"]
                name = tool["name"]
                row.pack(fill="x", pady=3)

                is_recog = tool in recognized
                status_label = tool.get("status_label")
                match_label = tool.get("match_label")
                name_label = tool.get("name_label")
                accent = tool.get("accent")

                if status_label is not None:
                    status_label.config(text="✓" if is_recog else "✗",
                                        fg=C_GREEN if is_recog else "#ef4444")
                if accent is not None:
                    accent.config(bg=C_GREEN if is_recog else "#3a4763")
                if name_label is not None:
                    name_label.config(fg=C_TEXT if is_recog else C_MUTED)

                if match_label is not None:
                    count = self.last_counts.get(name, 0)
                    if self.use_yolo:
                        # YOLO 模式：count 是置信度百分比
                        if is_recog and count > 0:
                            fg = (C_GREEN if count >= 70 else
                                  C_AMBER if count >= 50 else C_MUTED)
                            match_label.config(text=f"{count}%", fg=fg)
                        else:
                            match_label.config(text="--", fg="#4a5670")
                    else:
                        # SIFT 模式：count=内点数, thr=阈值, src=命中来源(强/弱/色)
                        thr = self.detect_threshold
                        src = self.last_source.get(name, "")
                        if is_recog:
                            if src == "色":
                                match_label.config(text=f"色 {count}", fg="#38bdf8")
                            elif src == "弱":
                                match_label.config(text=f"{count}/{thr}弱", fg=C_AMBER)
                            elif src == "强":
                                match_label.config(text=f"{count}/{thr}强", fg=C_GREEN)
                            else:
                                match_label.config(text=f"{count}/{thr}窗", fg=C_MUTED)
                        else:
                            match_label.config(
                                text=f"{count}/{thr}" if count > 0 else "--",
                                fg="#4a5670")

            total = len(self._valid_tools())
            self.count_badge.config(
                text=f"{len(recognized)}/{total}",
                fg=C_GREEN if recognized else C_MUTED)
            self._set_stats(f"当前识别 {len(recognized)} / {total} 个工具", C_TEXT)
            return len(recognized)
        except Exception as e:
            print(f"刷新列表出错: {e}")
            return 0

    # ---------------- 启停控制 ----------------

    def _valid_tools(self):
        """可用于检测/统计的工具：YOLO 模式全部有效；SIFT 模式需有模板特征(des)"""
        if self.use_yolo:
            return list(self.tools)
        return [t for t in self.tools if t.get("des") is not None]

    @staticmethod
    def _is_blank_frame(frame):
        """判断摄像头是否吐出了接近全黑/无信号的坏帧。"""
        if frame is None or getattr(frame, "size", 0) == 0:
            return True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return (float(gray.mean()) < BLANK_MEAN_THRESHOLD
                and float(gray.std()) < BLANK_STD_THRESHOLD)

    def _clear_realtime_detection_state(self):
        """清理实时检测缓存，避免旧框/旧结果挂在坏帧上。"""
        self.last_seen = {}
        self.last_counts = {}
        self.last_source = {}
        self._last_boxes = []
        self._box_store = {}
        self._last_annotated = None
        self._last_recog = None
        self._last_ui_refresh = 0.0
        self._last_speak = None
        self._has_detection_sample = False
        with self._frame_lock:
            self._latest_frame = None

    def _signal_lost_frame(self, reference=None):
        """生成一个无信号占位帧，替代黑屏叠旧框。"""
        if reference is not None and getattr(reference, "ndim", 0) >= 2:
            h, w = reference.shape[:2]
        else:
            h, w = 720, 1280
        frame = np.full((h, w, 3), (26, 14, 10), dtype=np.uint8)
        cv2.putText(
            frame,
            "Camera signal lost, retrying...",
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (248, 191, 56),
            2,
            cv2.LINE_AA,
        )
        return frame

    # 打开摄像头的尝试矩阵：按顺序逐个试，第一个出真画面的就用。
    # 排序原则：旧 USB 摄像头惯用的「DSHOW + 强设 720p + 不改格式」放最前，保证老设备零回退；
    # 后面几项专为 4K UVC 直播摄像机（如海康 DS-UVC-U168R）兜底——
    #   · 加 MJPG：4K/USB 机常常只有压缩流才出画面，不设就给原始大帧或干脆不出流；
    #   · 换 MSMF 后端：DSHOW 对现代 4K UVC 经常打不开或卡 4K 大帧，MSMF 往往更稳；
    #   · 放开分辨率（native=用摄像头默认值）：有些 4K 机拒绝被设成 720p，只好接受其原生分辨率。
    # (backend, 后端名, 是否强设720p, 是否设MJPG)
    # 分两档是为了"广度优先"（见 _open_camera）：
    #   _CAM_FAST 每个后端只留最可能成的一项，先把所有设备号快速扫一遍——
    #     摄像头飘到 1 号时能马上找到，不用先把 0 号的 6 种组合试穿（要多等好几秒）。
    #   _CAM_DEEP 是 4K UVC 的兜底格式，只对"确实打得开的设备号"再补试。
    _CAM_FAST = [
        (cv2.CAP_DSHOW, "DSHOW", True,  False),
        (cv2.CAP_MSMF,  "MSMF",  True,  False),
    ]
    _CAM_DEEP = [
        (cv2.CAP_DSHOW, "DSHOW", True,  True),
        (cv2.CAP_MSMF,  "MSMF",  True,  True),
        (cv2.CAP_MSMF,  "MSMF",  False, False),  # native 分辨率兜底
        (cv2.CAP_DSHOW, "DSHOW", False, False),
    ]
    _CAM_ATTEMPTS = _CAM_FAST + _CAM_DEEP   # 完整顺序（旧代码/脚本引用它）

    def _try_open(self, index, flag, force_720p, use_mjpg,
                  budget=1.2, accept_dark=False):
        """按指定后端/分辨率/格式打开一次，预热读帧判断是否有真画面。

        返回 (cap, opened, streamed)：
          - cap 非 None  → 拿到可用画面，已配好，可直接用；
          - opened       → 设备句柄能不能打开（区分「没这个设备」和「设备在但不出流」）；
          - streamed     → 有没有读到过真帧（哪怕全是黑的）。**这是新增的关键信号**：
                           读得到帧 = 摄像头是活的，只是自动曝光还没起来 / 镜头被挡，
                           不该当成"打不开"扔掉（这正是以前白白多试 5 个组合、
                           多花十几秒的根因）。

        budget：本次尝试的读帧时间预算（秒）。以前是"最多读 8 帧、每帧后 sleep(0.05)"，
        对慢启动的摄像头只给了约 0.4s，曝光还没起来就被判失败；现在改成按时间预算循环，
        读得到帧就不再 sleep，既不误杀慢启动的机器，也不会在被占用时干等。

        accept_dark=True：第二轮兜底用——只要读得到帧就接受，不再要求画面够亮。
        """
        cap = cv2.VideoCapture(index, flag)
        if not cap.isOpened():
            cap.release()
            return None, False, False

        # 关键：分辨率/格式必须在预热读帧之前设好。旧代码先用默认分辨率预热、之后才设 720p，
        # 4K 机就会拿 3840x2160 大帧预热，读帧奇慢——这正是换 4K 摄像头后"卡 + 打不开"的元凶之一。
        if use_mjpg:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
        if force_720p:
            # 1280x720：低分辨率会让远处/小工具像素不足而漏检，先保检测能力（之前降到 640x480 识别不出）。
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # 缓冲设为 1：丢掉积压旧帧，始终拿最新画面，消除"慢半拍"
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        streamed = False
        deadline = time.time() + budget
        while True:
            ret, frame = cap.read()
            if ret and frame is not None and frame.size:
                streamed = True
                # 隔行采样再算方差：4K native 那几种组合整帧是 2500 万个数，
                # 一秒预算里要算几十次会白烧 CPU；判"有没有真画面"用 1/64 的点足够。
                if accept_dark or float(frame[::8, ::8].std()) > 6.0:
                    return cap, True, True
            else:
                # 只有"读不到帧"才让出时间片；读得到就连着读，别人为拖慢开机
                time.sleep(0.03)
            if time.time() >= deadline:
                break

        cap.release()
        return None, True, streamed

    # ---- 上次成功配置的缓存：把整轮探测降成一次打开 ----

    def _load_cam_pref(self):
        """读上次成功打开摄像头的配置（读坏/不存在都当没有，绝不因此影响开机）。"""
        try:
            with open(CAMERA_PREF_PATH, "r", encoding="utf-8") as f:
                pref = json.load(f)
            return (int(pref["index"]), str(pref["backend"]),
                    bool(pref["force_720p"]), bool(pref["mjpg"]))
        except Exception:
            return None

    def _save_cam_pref(self, index, bname, force_720p, use_mjpg):
        try:
            with open(CAMERA_PREF_PATH, "w", encoding="utf-8") as f:
                json.dump({"index": index, "backend": bname,
                           "force_720p": force_720p, "mjpg": use_mjpg}, f)
        except Exception:
            pass

    def _open_camera(self, max_index=2):
        """打开一个可用摄像头，返回配置好的 VideoCapture（失败返回 None）。

        为什么这么写（"点一下要等 10 多秒"和"明明插着却说没摄像头"都是被下面这几条治掉的）：
          1. **优先用上次成功的配置**：命中就一次打开搞定（<1s），不再每次从头探整个矩阵。
          2. **广度优先**：先拿最可能成的「720p/默认格式」把 0→1→2 号全扫一遍（每号最多
             DSHOW/MSMF 两次打开），再回头给开得开的设备号试 MJPG/native 那几种 4K 兜底组合。
             以前是"把 0 号 6 种组合试穿再看 1 号"，0 号被占用时要白等 7 秒才轮到真摄像头。
          3. **绝不在 0 号就放弃**：日志里实测有过「0 号全后端打不开、真摄像头在 1 号」的机器
             （USB 摄像头枚举在 Windows 上会飘），旧逻辑直接报"未找到摄像头"。实测探一个
             不存在的设备号只要 0~40ms（不是注释里说的 7-9s），扫完 1、2 号也就多花 0.1s。
          4. **判"有画面"的标准别太急**：见 _try_open——读得到帧就说明设备是活的，
             只是曝光没起来/环境暗，直接接受，不该关掉再试下一套（这是那 10 多秒的主因）。
          5. **总预算兜底**：最坏情况（几个设备号都能打开却都不出流）也不会无限拖，
             超过 CAMERA_PROBE_BUDGET 就停下来给明确提示。
        """
        # 失败原因：None=还没结论 / "busy"=能打开但读不到真画面(疑似被占用) / "none"=根本没摄像头。
        self._cam_fail_reason = None
        t_start = time.time()

        def over_budget():
            return time.time() - t_start > CAMERA_PROBE_BUDGET

        # ---- 0. 先试上次成功的配置 ----
        pref = self._load_cam_pref()
        if pref is not None:
            index, bname, force_720p, use_mjpg = pref
            flag = cv2.CAP_DSHOW if bname == "DSHOW" else cv2.CAP_MSMF
            cap, _, _ = self._try_open(index, flag, force_720p, use_mjpg,
                                       budget=1.5, accept_dark=True)
            if cap is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                diag(f"摄像头：命中上次配置 设备{index} {bname} "
                     f"{'720p' if force_720p else 'native'}/{'MJPG' if use_mjpg else '默认'}"
                     f" → ✅ {w}x{h}，耗时 {(time.time() - t_start) * 1000:.0f}ms")
                return cap
            # 删掉失效缓存：否则下次开机还要先在这套配置上白等 1.5s 才转完整探测
            # （探测成功会重新写入，所以只是"临时抽风"的话下一次就自动补回来）
            try:
                os.remove(CAMERA_PREF_PATH)
            except OSError:
                pass
            diag("摄像头：上次配置不再可用（已清掉缓存），转入完整探测")

        # ---- 1. 完整探测：先广度(_CAM_FAST 扫所有设备号)，再深度(_CAM_DEEP 补格式) ----
        opened_any = {}     # 设备号 -> 该号至少被某个后端打开过（说明设备存在）
        dead = set()        # (设备号, 后端名)：连打开都失败的组合，后面不再试
        dark_hit = None     # (index, flag, bname, force_720p, use_mjpg)：读得到帧但画面偏暗

        def sweep(attempts, indices):
            """按给定组合表扫一遍给定设备号；拿到画面就直接返回 cap。"""
            nonlocal dark_hit
            for index in indices:
                for flag, bname, force_720p, use_mjpg in attempts:
                    if (index, bname) in dead or over_budget():
                        continue
                    t0 = time.time()
                    cap, opened, streamed = self._try_open(index, flag, force_720p, use_mjpg)
                    res = f"{'720p' if force_720p else 'native'}/{'MJPG' if use_mjpg else '默认'}"
                    cost = (time.time() - t0) * 1000
                    if opened:
                        opened_any[index] = True
                    if cap is not None:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        diag(f"摄像头探测：设备{index} {bname} {res} → ✅ 打开成功 {w}x{h}，"
                             f"耗时 {cost:.0f}ms，总 {(time.time() - t_start) * 1000:.0f}ms")
                        self._save_cam_pref(index, bname, force_720p, use_mjpg)
                        return cap
                    if not opened:
                        dead.add((index, bname))   # 同设备同后端换格式也开不了，别白试
                        why = "连打开都失败"
                    elif streamed:
                        # 短路：读得到帧就说明这套组合是通的，只是画面暗（暗房/镜头被挡）。
                        # 换 MJPG/MSMF/native 不会让画面变亮，继续试纯属白等 → 交给兜底。
                        why = "能出流但画面偏暗(曝光未起/遮挡)"
                        dark_hit = (index, flag, bname, force_720p, use_mjpg)
                    else:
                        why = "能打开但读不到帧(忙/格式不出流)"
                    diag(f"摄像头探测：设备{index} {bname} {res} → ✗ {why}，耗时 {cost:.0f}ms")
                    if dark_hit is not None:
                        return None
            return None

        cap = sweep(self._CAM_FAST, range(max_index + 1))
        if cap is not None:
            return cap
        if dark_hit is None and not over_budget():
            # 深度补探只针对"确实存在的设备号"（4K UVC 常要 MJPG / native 才出流）
            cap = sweep(self._CAM_DEEP, sorted(opened_any))
            if cap is not None:
                return cap

        # ---- 2. 兜底：有组合读得到帧、只是不够亮 → 直接用它（摄像头是活的，别丢） ----
        if dark_hit is not None:
            index, flag, bname, force_720p, use_mjpg = dark_hit
            # 先给 MJPG 变体一次机会再认命：有些 4K UVC 不压缩时只吐无效/暗帧，
            # 换 MJPG 才出真画面。只多花一次尝试（~1s），却能救回一整类 4K 机。
            if not use_mjpg:
                cap, _, _ = self._try_open(index, flag, force_720p, True)
                if cap is not None:
                    diag(f"摄像头：设备{index} {bname} 默认格式只出暗帧，改 MJPG → ✅ 总耗时 "
                         f"{(time.time() - t_start) * 1000:.0f}ms")
                    self._save_cam_pref(index, bname, force_720p, True)
                    return cap
            cap, _, _ = self._try_open(index, flag, force_720p, use_mjpg,
                                       budget=2.0, accept_dark=True)
            if cap is not None:
                diag(f"摄像头：按'能出流但偏暗'兜底采用 设备{index} {bname} → ✅ 总耗时 "
                     f"{(time.time() - t_start) * 1000:.0f}ms（画面暗请检查环境光/镜头盖）")
                self._save_cam_pref(index, bname, force_720p, use_mjpg)
                return cap

        if over_budget():
            diag(f"摄像头探测：超过 {CAMERA_PROBE_BUDGET}s 预算，停止探测")
        if opened_any:
            self._cam_fail_reason = "busy"
            diag("⚠️ 摄像头疑似被其它程序占用（请关闭 oCam/微信视频/浏览器/相机后重试）")
        else:
            self._cam_fail_reason = "none"
            diag("⚠️ 未找到可用摄像头，回退到选图片模式")
        return None

    def _check_ready(self):
        """检测前置检查：检测器就绪 + 有可用工具。不通过弹框并返回 False。"""
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
        """清掉上一轮残留的帧/框，避免新一轮开头闪到旧画面。"""
        self._last_boxes = []
        self._box_store = {}
        self._last_annotated = None
        self._camera_signal_lost = False
        with self._frame_lock:
            self._latest_frame = None
            self._display_frame = None

    def _begin_realtime(self, status_text):
        """启动实时模式（摄像头/视频共用）：读帧线程 + 主线程 UI 泵。"""
        self.running = True
        self.should_speak = True
        self._reset_frame_state()
        self._set_buttons_running(True)
        self._set_stats(status_text, C_GREEN)
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        self._start_ui_pump()

    def detect_camera(self):
        """摄像头检测：打开摄像头 → 实时双线程检测。"""
        if self.opening_camera:
            return
        if self.running:
            self.stop_detection()
        if not self._check_ready():
            return

        self.opening_camera = True
        self._camera_open_token += 1
        token = self._camera_open_token
        self.use_camera = True
        self.is_video_file = False
        self.static_frame = None
        self._reset_frame_state()
        self._set_buttons_running(True)
        self.video_label.configure(image="", text="📷  正在打开摄像头...")
        self.video_label.imgtk = None
        self._imgtk = None
        self._set_stats("正在打开摄像头...", C_AMBER)
        threading.Thread(target=self._open_camera_worker, args=(token,), daemon=True).start()

    def _open_camera_worker(self, token):
        """后台打开摄像头，避免 OpenCV 探测阻塞 Tk 主线程。"""
        try:
            cap = self._open_camera()
            error = None
        except Exception as e:
            cap = None
            error = e
        try:
            self.root.after(0, lambda: self._on_camera_opened(token, cap, error))
        except Exception:
            if cap is not None:
                cap.release()

    def _on_camera_opened(self, token, cap, error=None):
        """摄像头打开结果回到主线程处理。"""
        if token != self._camera_open_token or not self.opening_camera:
            if cap is not None:
                cap.release()
            return

        self.opening_camera = False
        if error is not None:
            if cap is not None:
                cap.release()
            self._set_buttons_running(False)
            self._set_stats("摄像头打开失败", "#ff6b6b")
            messagebox.showerror("摄像头打开失败", f"打开摄像头时出错：\n{error}")
            return

        if cap is None:
            self._set_buttons_running(False)
            self.video_label.configure(image="", text="📷  点击下方按钮开始检测")
            self.video_label.imgtk = None
            self._imgtk = None
            self._set_stats("摄像头不可用", "#ff6b6b")
            if getattr(self, "_cam_fail_reason", None) == "busy":
                # 能打开但读不到画面 = 设备在、被独占。明确告诉用户关掉占用程序，别让他以为是程序坏了。
                messagebox.showwarning(
                    "摄像头被占用",
                    "摄像头能打开、但读不到画面，多半是被其它程序占用了。\n\n"
                    "请先关闭这些程序后再点检测：\n"
                    "  • oCam / 录屏软件\n"
                    "  • 微信 / QQ / 腾讯会议 的视频\n"
                    "  • 浏览器里开着摄像头的网页\n"
                    "  • Windows「相机」App\n\n"
                    "关掉后再点「📷 摄像头检测」即可秒开；也可改用「🎬 视频检测」「🖼 图片检测」。")
            else:
                messagebox.showwarning(
                    "未找到摄像头",
                    "没有检测到可用摄像头。\n请检查摄像头是否插好，"
                    "或改用「🎬 视频检测」「🖼 图片检测」。")
            return

        self.cap = cap
        self.use_camera = True
        self.is_video_file = False
        self.static_frame = None
        self._begin_realtime("正在检测（摄像头实时）…")

    def detect_image(self):
        """图片检测：选一张图片 → 检测一次。"""
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
        self._reset_frame_state()
        self._set_buttons_running(True)
        self._set_stats("检测图片中…", C_AMBER)
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()   # 静态图一次性显示，不需要 UI 泵

    def open_video(self):
        """选一个视频文件，按原生帧率播放并实时检测（复用摄像头双线程逻辑）。

        与摄像头的区别：① 视频读帧不会自带节奏，要按视频 fps 主动限速，否则一闪而过；
        ② 放到结尾要自动停。这两点在 _camera_loop 里按 is_video_file 处理。
        """
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
        self.use_camera = True       # 走实时双线程循环（读帧+后台推理）
        self.is_video_file = True
        self.static_frame = None
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_fps = fps if fps and fps > 1 else 25.0

        self._begin_realtime(f"正在检测视频（{os.path.basename(file_path)}）…")

    def stop_detection(self):
        was_opening = self.opening_camera
        was_signal_lost = self._camera_signal_lost
        self.opening_camera = False
        self._camera_open_token += 1
        self.running = False
        self.should_speak = False
        self._infer_running = False   # 停掉后台推理线程
        self._camera_signal_lost = False
        self.voice.stop()             # 立刻打断正在念的语音（不管念没念完）

        if self.cap is not None:
            cap = self.cap
            self.cap = None
            cap.release()

        with self._frame_lock:
            self._latest_frame = None
            if was_signal_lost or was_opening:
                self._display_frame = None
                self._display_id += 1

        self._set_buttons_running(False)
        self.refresh_visible_list()
        if was_signal_lost or was_opening:
            self.video_label.configure(image="", text="📷  点击下方按钮开始检测")
            self.video_label.imgtk = None
            self._imgtk = None
        self._set_stats("已停止", "#ff6b6b")

    def reset_display(self):
        """清空显示 = 停止当前检测 + 清空 last_seen + 清空画面。

        必须先停检测，否则静态/摄像头线程会立刻把 last_seen 填回去，看起来"没清掉"。
        """
        if self.running or self.opening_camera:
            self.stop_detection()

        self.last_seen = {}
        self.last_counts = {}
        self._last_boxes = []
        self._last_annotated = None
        self._last_speak = None
        self._has_detection_sample = False
        try:
            self.refresh_visible_list()
            self.video_label.configure(image="", text="📷  点击下方按钮开始检测")
            self.video_label.imgtk = None
            self._imgtk = None
            self._set_stats("已清空，待检测", C_MUTED)
        except Exception as e:
            print(f"清空显示出错: {e}")

    # ---------------- 检测循环 ----------------

    @staticmethod
    def _letterbox(frame_bgr):
        """等比缩放并居中贴到固定尺寸画布（不足处补视频区同色黑边）。

        固定尺寸是防闪的前提：尺寸一变，Tk 就得重算布局、擦背景重画，
        画面和右侧清单都会跟着抖一下。720p 摄像头缩放后正好铺满，看不到黑边。
        """
        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return np.full((DISPLAY_H, DISPLAY_W, 3), CANVAS_BG_BGR, np.uint8)
        scale = min(DISPLAY_W / w, DISPLAY_H / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=interp)
        if nw == DISPLAY_W and nh == DISPLAY_H:
            return resized
        canvas = np.full((DISPLAY_H, DISPLAY_W, 3), CANVAS_BG_BGR, np.uint8)
        y0, x0 = (DISPLAY_H - nh) // 2, (DISPLAY_W - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

    def _set_stats(self, text, color):
        """状态栏文字：内容没变就不重设，避免 UI 泵每 15ms 触发一次无谓重绘。"""
        if getattr(self, "_stats_cache", None) == (text, color):
            return
        self._stats_cache = (text, color)
        self.stats_label.config(text=text, fg=color)

    def _show_frame(self, frame_bgr):
        """把 BGR 帧渲染到视频区（原地更新像素，不重建控件图片 → 不闪）。

        闪屏的另一半原因在这里：旧写法每帧都 new 一个 PhotoImage 再 configure(image=...)，
        Tk 每次都要擦掉标签背景、重算尺寸再贴图，几十 Hz 下就是肉眼可见的一闪一闪。
        改成"画布尺寸固定 + PhotoImage 只建一次 + paste() 原地覆盖像素"后，
        既不 configure 也不重算布局，画面平顺。
        """
        canvas = self._letterbox(frame_bgr)
        img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        if self._imgtk is None:
            self._imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=self._imgtk, text="")
            self.video_label.imgtk = self._imgtk
        else:
            self._imgtk.paste(img)      # 原地换像素，控件完全不动

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
        """复位 UI 泵状态并在主线程启动它（由主线程的按钮回调调用）。"""
        self._last_shown_id = -1
        self._last_recog = None
        self._last_ui_refresh = 0.0
        self._last_speak = None
        self._has_detection_sample = False
        with self._frame_lock:
            self._display_frame = None
            self._display_id = 0
        self.root.after(0, self._ui_pump)

    def _camera_loop(self):
        """读帧线程：持续读帧 → 叠最近的框（纯 cv2，不碰 Tkinter）→ 存成待显示帧。

        三方分工：本线程只读帧+叠框（不碰 Tkinter）；_inference_worker 后台跑推理回写框；
        主线程 _ui_pump 取待显示帧画图+刷列表。Tkinter 全在主线程，避免黑屏闪烁。
        框常驻：每帧把推理线程最新回写的 _last_boxes 叠上（render_live 缓存贴图，几乎零开销），
        所以框跟着实时画面、不闪。
        """
        self._box_store = {}  # 清掉上一轮的框存活记录
        # 启动后台推理线程
        self._infer_running = True
        worker = threading.Thread(target=self._inference_worker, daemon=True)
        worker.start()

        # 视频文件：按原生帧率播放（否则读多快放多快，一闪而过）
        frame_interval = (1.0 / self.video_fps) if self.is_video_file else 0.0
        next_frame_t = time.time()

        # 诊断计数（读帧帧率）
        diag_t0 = time.time()
        diag_frames = 0
        bad_frames = 0       # 连续"读帧失败"计数
        blank_frames = 0     # 连续"读到但全黑"计数（和读帧失败分开算，见下）
        last_bad_diag = 0.0
        # 开流宽限期：摄像头刚打开时自动曝光还没起来，头几十帧就是黑的。
        # 这段时间内不判"信号丢失"，否则一点开就闪一下橙色"无信号"再跳回画面 —— 就是"闪屏"。
        warmup_until = time.time() + CAMERA_WARMUP_GRACE
        while self.running and self.cap is not None:
            cap = self.cap
            if cap is None:
                break
            ret, frame = cap.read()
            if not self.running or self.cap is None:
                break
            if not ret or frame is None:
                if self.is_video_file:
                    break          # 视频放完 → 退出循环，自动停止
                bad_frames += 1
                if bad_frames >= BAD_FRAME_LIMIT and time.time() >= warmup_until:
                    # 只在"刚进入丢失状态"那一次做重活。否则读帧失败期间每轮都要重建
                    # 占位帧 + 清空识别状态，而 _clear_realtime_detection_state 会把
                    # _last_ui_refresh 归零、把 UI 泵的 1s 节流打穿 —— 右侧清单会被
                    # 每 15ms 重排一次，本身就是一种闪。
                    if not self._camera_signal_lost:
                        self._camera_signal_lost = True
                        self._clear_realtime_detection_state()
                        with self._frame_lock:
                            self._display_frame = self._signal_lost_frame()
                            self._display_id += 1
                    now_bad = time.time()
                    if now_bad - last_bad_diag >= 2.0:
                        diag(f"摄像头读帧失败连续 {bad_frames} 次：已清理旧框，等待恢复")
                        last_bad_diag = now_bad
                time.sleep(0.05)   # 摄像头偶发读帧失败 → 重试
                continue

            if not self.is_video_file and self._is_blank_frame(frame):
                # 全黑帧 ≠ 断流：曝光没起来、镜头被挡、灯灭了都会全黑，但流是好的。
                # 所以① 用独立的、更大的阈值 BLANK_FRAME_LIMIT（≈1s 以上）；
                #     ② 开流宽限期内一律不报。这两条一起消掉"刚打开闪一下无信号"的闪屏。
                blank_frames += 1
                if blank_frames >= BLANK_FRAME_LIMIT and time.time() >= warmup_until:
                    if not self._camera_signal_lost:   # 同上：进入丢失状态只做一次
                        self._camera_signal_lost = True
                        self._clear_realtime_detection_state()
                        with self._frame_lock:
                            self._display_frame = self._signal_lost_frame(frame)
                            self._display_id += 1
                    now_bad = time.time()
                    if now_bad - last_bad_diag >= 2.0:
                        diag(f"摄像头输出黑帧连续 {blank_frames} 次：已清理旧框，等待恢复")
                        last_bad_diag = now_bad
                else:
                    # 还没到判定线：照常把这帧显示出去，保持画面连续（黑就黑，别闪占位图）
                    with self._frame_lock:
                        self._latest_frame = frame
                        self._display_frame = frame
                        self._display_id += 1
                diag_frames += 1   # 黑帧也算读到了帧，否则日志里帧率会假摔成 0.x fps
                time.sleep(0.02)
                continue

            if self._camera_signal_lost:
                diag("摄像头画面已恢复")
            bad_frames = 0
            blank_frames = 0
            self._camera_signal_lost = False

            now = time.time()
            diag_frames += 1

            # 把最新帧交给推理线程（只存引用，worker 取用时自己 copy）
            with self._frame_lock:
                self._latest_frame = frame

            # 叠框（纯 cv2/numpy，不碰 Tkinter）。每帧把推理线程最新回写的框叠上 → 框常驻不闪
            if self.use_yolo:
                display = (self.detector_engine.render_live(frame.copy(), self._last_boxes)
                           if self._last_boxes else frame)
            else:
                display = self._last_annotated if self._last_annotated is not None else frame

            # 存成"待显示帧"，交给主线程 _ui_pump 去画。绝不在这个后台线程里调 Tkinter——
            # 后台线程直接建 PhotoImage/configure 会和主线程重绘抢，导致画面间歇性闪黑。
            with self._frame_lock:
                self._display_frame = display
                self._display_id += 1

            # 每 2 秒报一次读帧帧率（写进 run_diag.txt 便于远程诊断）
            if now - diag_t0 >= 2.0:
                fps = diag_frames / (now - diag_t0)
                hit = len([n for n, t in list(self.last_seen.items())
                           if now - t < VISIBILITY_WINDOW])
                diag(f"读帧线程：{fps:.1f} fps，窗口内命中 {hit} 个")
                diag_t0 = now
                diag_frames = 0

            # 限速：视频文件按原生帧率播放；摄像头只让出极短时间
            if self.is_video_file:
                next_frame_t += frame_interval
                delay = next_frame_t - time.time()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_frame_t = time.time()   # 落后了就别越积越多
            else:
                time.sleep(0.01)

        # 读帧线程退出 → 通知推理线程收尾
        self._infer_running = False

        # 视频自然放完（非手动停止）→ 回主线程复位按钮与状态
        if self.is_video_file and self.running:
            self.root.after(0, self._on_video_finished)

    def _ui_pump(self):
        """主线程 UI 泵：定时取最新"待显示帧"来画 + 刷列表 + 触发语音。

        所有 Tkinter 操作都在这里（主线程）完成，跟 mainloop 重绘同步，
        避免后台线程直接画图导致的黑屏闪烁。running=False 时自动停止重调度。
        """
        if not self.running:
            return
        now = time.time()

        # 只在有新帧时才画（避免重复建 PhotoImage）
        with self._frame_lock:
            frame = self._display_frame
            did = self._display_id
        if frame is not None and did != self._last_shown_id:
            self._show_frame(frame)
            self._last_shown_id = did

        if self._camera_signal_lost:
            if now - self._last_ui_refresh >= 1.0:
                self.refresh_visible_list()
                self._last_ui_refresh = now
            self._set_stats("摄像头无画面，正在重试...", C_AMBER)
            self.root.after(15, self._ui_pump)
            return

        # 列表刷新：识别集合变化就刷（节流 1s），否则每 UI_REFRESH_INTERVAL 兜底
        recog_now = frozenset(n for n, t in list(self.last_seen.items())
                              if now - t < VISIBILITY_WINDOW)
        if ((recog_now != self._last_recog and now - self._last_ui_refresh >= 1.0)
                or now - self._last_ui_refresh >= UI_REFRESH_INTERVAL):
            self.refresh_visible_list()
            self._last_ui_refresh = now
            self._last_recog = recog_now

        # 语音播报：必须等模型至少完成过一次检测，再从检测结果出来后计时。
        if (self.should_speak and self._has_detection_sample
                and self._last_speak is not None
                and now - self._last_speak >= SPEAK_INTERVAL):
            self.speak_missing()
            self._last_speak = now

        self.root.after(15, self._ui_pump)   # ~每 15ms 一次，跟随主循环重绘

    def _on_video_finished(self):
        """视频播放结束后在主线程里复位 UI（手动停止不走这里）。"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._set_buttons_running(False)
        self.refresh_visible_list()
        self._set_stats("视频检测完毕", "#00ff88")

    def _hold_boxes(self, boxes):
        """框防闪：每个工具的框按名字保留 VISIBILITY_WINDOW 秒。

        阈值边缘的工具会一会儿检到、一会儿漏掉，若每次推理直接用当次结果，框就一闪一闪。
        这里把"最近一次检到的框"按工具名存活一段时间：偶尔漏检不立刻抹掉、沿用上次位置，
        直到真的超过存活期才消失。返回当前应显示的框列表（每个工具一个，取最新位置）。
        只有推理线程调用，self._box_store 不跨线程写，赋值给 _last_boxes 是原子操作。
        """
        now = time.time()
        for b in boxes:
            self._box_store[b[4]] = (b, now)   # b[4] 是工具中文名
        held = []
        for name in list(self._box_store.keys()):
            b, ts = self._box_store[name]
            if now - ts < VISIBILITY_WINDOW:
                held.append(b)
            else:
                del self._box_store[name]       # 过期 → 框消失
        return held

    def _inference_worker(self):
        """后台推理线程：每 DETECT_INTERVAL 取一次最新帧跑 YOLO/SIFT，
        只回写 _last_boxes / last_seen / last_counts，不碰 Tkinter（线程安全）。

        与显示线程解耦：推理慢只拖慢"框刷新率"，绝不阻塞显示线程读帧/显示。
        """
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
            if frame is None or self._camera_signal_lost:
                time.sleep(0.01)
                continue
            last_detect = now

            t_infer = time.time()
            inference_ok = True
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
                inference_ok = False
            if self._camera_signal_lost:
                self._last_boxes = []
                self._last_annotated = None
                continue
            diag_infer_ms += (time.time() - t_infer) * 1000
            diag_infers += 1

            stamp = time.time()
            for name in detected:
                self.last_seen[name] = stamp
            self.last_counts.update(counts)
            if inference_ok and not self._has_detection_sample:
                self._has_detection_sample = True
                self._last_speak = stamp
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
        self._show_frame(display)
        recognized_count = self.refresh_visible_list()
        if self.should_speak:
            self.speak_missing()

        self.running = False
        try:
            self._set_buttons_running(False)
            total = len(self._valid_tools())
            # 底部数字必须和上方列表一致——用 refresh 返回的 recognized_count，
            # 而不是本帧 len(detected)；连续多次点检测时窗口里旧条目仍算
            self._set_stats(f"检测完成：识别 {recognized_count}/{total}", "#00ff88")
        except Exception:
            pass

    # ---------------- 语音 ----------------

    def speak_missing(self):
        """播报当前还缺失的工具（基于可见性窗口）"""
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
