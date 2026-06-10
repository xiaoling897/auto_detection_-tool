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
  - 每 8 秒播报一次当前缺失的工具
  - "🔄 清空显示"先停检测再清空 last_seen

依赖 data/smart_tools.json 里的工具配置 + data/smart_templates/ 下的模板图。
"""
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from .detector import ToolDetector
from .loader import load_tools, load_tool_names
from .utils import DATA_DIR, imread_unicode
from .voice import VoiceReporter

UI_REFRESH_INTERVAL = 3.0   # 列表 UI 刷新节奏
DETECT_INTERVAL = 0.4       # 检测限频：每 N 秒推理一次（其余时间只读帧+叠最近的框，保证视频顺）
VISIBILITY_WINDOW = 3.0     # 工具可见性窗口：最近 N 秒内识别到才显示
SPEAK_INTERVAL = 8.0        # 语音播报间隔

# 诊断日志：写到 exe 同级 data/run_diag.txt（windowed 包没控制台也能留痕），同时打印。
# 排查"卡 + 识别不出"用——记录后端、单次推理耗时、检测速率、显示帧率、读帧耗时。
DIAG_PATH = DATA_DIR / "run_diag.txt"


def diag(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    try:
        with open(DIAG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


class ToolDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智能工具检测系统")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1a1a2e")

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
        self._last_speak = 0.0

        diag("=" * 50)
        diag(f"启动诊断：后端={'YOLO' if self.use_yolo else 'SIFT(回退)'}, "
             f"cpu_count={os.cpu_count()}, 工具数={len(self.tools)}, "
             f"加载错误={self.model_error or '无'}")

        self.create_ui()

    # ---------------- UI ----------------

    def create_ui(self):
        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill="both", expand=True)

        # 左侧 - 视频
        left_frame = tk.Frame(main_frame, bg="#1a1a2e")
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        title_label = tk.Label(left_frame, text="🔧 智能工具检测系统",
                               font=("Microsoft YaHei UI", 28, "bold"),
                               bg="#1a1a2e", fg="#00d4ff")
        title_label.pack(pady=15)

        video_container = tk.Frame(left_frame, bg="#0d1117", relief="ridge", bd=2)
        video_container.pack(fill="both", expand=True, pady=10)

        self.video_label = tk.Label(video_container, bg="#0d1117", text="点击立即检测")
        self.video_label.pack(fill="both", expand=True, padx=5, pady=5)

        # 按钮
        button_frame = tk.Frame(left_frame, bg="#1a1a2e")
        button_frame.pack(pady=20)

        self.start_btn = tk.Button(button_frame, text="▶ 立即检测",
                                   font=("Microsoft YaHei UI", 16, "bold"),
                                   bg="#238636", fg="white", width=14, height=2,
                                   relief="flat", cursor="hand2",
                                   command=self.start_detection)
        self.start_btn.pack(side="left", padx=15)

        self.stop_btn = tk.Button(button_frame, text="⏹ 停止检测",
                                  font=("Microsoft YaHei UI", 16, "bold"),
                                  bg="#da3633", fg="white", width=14, height=2,
                                  relief="flat", cursor="hand2",
                                  command=self.stop_detection, state="disabled")
        self.stop_btn.pack(side="left", padx=15)

        self.video_btn = tk.Button(button_frame, text="🎬 打开视频",
                                   font=("Microsoft YaHei UI", 16, "bold"),
                                   bg="#1f6feb", fg="white", width=14, height=2,
                                   relief="flat", cursor="hand2",
                                   command=self.open_video)
        self.video_btn.pack(side="left", padx=15)

        self.reset_btn = tk.Button(button_frame, text="🔄 清空显示",
                                   font=("Microsoft YaHei UI", 16, "bold"),
                                   bg="#9333ea", fg="white", width=14, height=2,
                                   relief="flat", cursor="hand2",
                                   command=self.reset_display)
        self.reset_btn.pack(side="left", padx=15)

        # 右侧 - 工具列表
        right_frame = tk.Frame(main_frame, bg="#16213e", width=350)
        right_frame.pack(side="right", fill="y", padx=10, pady=10)

        list_title = tk.Label(right_frame, text="📋 已识别工具",
                              font=("Microsoft YaHei UI", 16, "bold"),
                              bg="#16213e", fg="#00ff88")
        list_title.pack(pady=20)

        self.tool_status_frame = tk.Frame(right_frame, bg="#16213e")
        self.tool_status_frame.pack(fill="both", expand=True, padx=15, pady=10)

        self.build_tool_rows()

        self.stats_label = tk.Label(right_frame, text="待检测...",
                                    font=("Microsoft YaHei UI", 12),
                                    bg="#16213e", fg="#ffffff")
        self.stats_label.pack(pady=20)

        # 初始状态：所有工具显示为未识别（✗ 灰色）——必须放在 stats_label 创建之后
        self.refresh_visible_list()

    def build_tool_rows(self):
        """为每个工具预创建一行 widget。refresh_visible_list 决定 pack 顺序 + 状态颜色。"""
        for tool in self.tools:
            name = tool["name"]
            row = tk.Frame(self.tool_status_frame, bg="#16213e")

            name_label = tk.Label(row, text=name,
                                  font=("Microsoft YaHei UI", 11),
                                  bg="#16213e", fg="#ffffff", anchor="w")
            name_label.pack(side="left", fill="x", expand=True)

            match_label = tk.Label(row, text="--",
                                   font=("Microsoft YaHei UI", 10),
                                   bg="#16213e", fg="#888888", width=11)
            match_label.pack(side="right", padx=5)

            status_indicator = tk.Label(row, text="✗",
                                        font=("Arial", 14),
                                        bg="#16213e", fg="#ff6b6b", width=3)
            status_indicator.pack(side="right")

            tool["row"] = row
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
                row.pack(fill="x", pady=5)

                is_recog = tool in recognized
                status_label = tool.get("status_label")
                match_label = tool.get("match_label")
                name_label = tool.get("name_label")

                if status_label is not None:
                    if is_recog:
                        status_label.config(text="✓", fg="#00ff88")
                    else:
                        status_label.config(text="✗", fg="#ff6b6b")

                if name_label is not None:
                    name_label.config(fg="#ffffff" if is_recog else "#888888")

                if match_label is not None:
                    count = self.last_counts.get(name, 0)
                    if self.use_yolo:
                        # YOLO 模式：count 是置信度百分比
                        if is_recog and count > 0:
                            fg = ("#00ff88" if count >= 70 else
                                  "#ffcc00" if count >= 50 else "#888888")
                            match_label.config(text=f"{count}%", fg=fg)
                        else:
                            match_label.config(text="--", fg="#555555")
                    else:
                        # SIFT 模式：count=内点数, thr=阈值, src=命中来源(强/弱/色)
                        thr = self.detect_threshold
                        src = self.last_source.get(name, "")
                        if is_recog:
                            if src == "色":
                                match_label.config(text=f"色 {count}", fg="#00aaff")
                            elif src == "弱":
                                match_label.config(text=f"{count}/{thr} 弱", fg="#ffcc00")
                            elif src == "强":
                                match_label.config(text=f"{count}/{thr} 强", fg="#00ff88")
                            else:
                                match_label.config(text=f"{count}/{thr} 窗", fg="#888888")
                        else:
                            match_label.config(
                                text=f"{count}/{thr}" if count > 0 else "--",
                                fg="#555555")

            total = len(self._valid_tools())
            self.stats_label.config(
                text=f"当前识别: {len(recognized)}/{total}",
                fg="#ffffff")
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

    # 打开摄像头的尝试矩阵：按顺序逐个试，第一个出真画面的就用。
    # 排序原则：旧 USB 摄像头惯用的「DSHOW + 强设 720p + 不改格式」放最前，保证老设备零回退；
    # 后面几项专为 4K UVC 直播摄像机（如海康 DS-UVC-U168R）兜底——
    #   · 加 MJPG：4K/USB 机常常只有压缩流才出画面，不设就给原始大帧或干脆不出流；
    #   · 换 MSMF 后端：DSHOW 对现代 4K UVC 经常打不开或卡 4K 大帧，MSMF 往往更稳；
    #   · 放开分辨率（native=用摄像头默认值）：有些 4K 机拒绝被设成 720p，只好接受其原生分辨率。
    # (backend, 后端名, 是否强设720p, 是否设MJPG)
    _CAM_ATTEMPTS = [
        (cv2.CAP_DSHOW, "DSHOW", True,  False),
        (cv2.CAP_DSHOW, "DSHOW", True,  True),
        (cv2.CAP_MSMF,  "MSMF",  True,  False),
        (cv2.CAP_MSMF,  "MSMF",  True,  True),
        (cv2.CAP_MSMF,  "MSMF",  False, False),  # native 分辨率兜底
        (cv2.CAP_DSHOW, "DSHOW", False, False),
    ]

    def _try_open(self, index, flag, force_720p, use_mjpg, warmup_frames):
        """按指定后端/分辨率/格式打开一次，预热读帧判断是否有真画面。

        返回 (cap 或 None, opened)：
          - cap 非 None  → 出了真画面，已配好，可直接用；
          - cap 为 None 且 opened=True  → 设备能打开但没出可用画面（黑/灰屏或出流失败）；
          - cap 为 None 且 opened=False → 这个设备号/后端根本打不开。
        opened 用于上层判断「0 号是否存在摄像头子系统」，决定要不要继续往后探。
        """
        cap = cv2.VideoCapture(index, flag)
        if not cap.isOpened():
            cap.release()
            return None, False

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

        for _ in range(warmup_frames):
            ret, frame = cap.read()
            if ret and frame is not None and float(frame.std()) > 6.0:
                return cap, True
            time.sleep(0.05)

        cap.release()
        return None, True

    def _open_camera(self, max_index=2, warmup_frames=8):
        """打开一个可用摄像头，返回配置好的 VideoCapture（失败返回 None）。

        换摄像头也能用——不写死设备号、不写死后端、不靠单帧判断：
          - 逐个设备号（0→max_index，外接 USB 常在 1、2）× 逐项尝试矩阵 _CAM_ATTEMPTS
            （DSHOW/MSMF 双后端 + 720p/native 双分辨率 + 选配 MJPG），第一个出真画面的就用。
          - 关键早退出：若 0 号在**所有后端**都连打开都失败，说明基本没有摄像头子系统，直接放弃，
            不再往后探。因为 DSHOW 探测不存在的设备号每个要卡 7-9s，无摄像头的机器否则要白等。
          - 预热读最多 warmup_frames 帧：很多摄像头首帧是黑/绿/花屏，读到方差够大（std>6.0，
            能区分真画面与均匀灰屏的虚拟设备）就判定可用，避免把预热慢的真摄像头误判成无摄像头。
        """
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
                # 失败也记日志（写进 run_diag.txt），方便在接了摄像头那台机器远程定位卡在哪
                why = "连打开都失败" if not opened else "能打开但读不到真画面(忙/格式不出流)"
                diag(f"摄像头探测：设备{index} {bname} {res} → ✗ {why}")
            # 0 号所有后端都打不开 → 基本无摄像头，立即放弃，避免后续设备号的 7-9s 空卡
            if index == 0 and not opened_any:
                diag("摄像头探测：0号所有后端都打不开 → 判定无摄像头子系统，停止探测")
                break

        diag("⚠️ 未找到可用摄像头，回退到选图片模式")
        return None

    def start_detection(self):
        if self.detector_engine is None:
            msg = (
                "检测器未就绪，无法开始检测。\n\n"
                f"详细错误：\n{self.model_error or '未知'}"
            )
            messagebox.showwarning("检测器未就绪", msg)
            return
        valid = self._valid_tools()
        if not valid:
            messagebox.showwarning(
                "警告",
                "没有任何可用工具。YOLO 模式请检查 data/yolo/best.pt 和 class_map.json；"
                "SIFT 模式请检查 data/smart_tools.json 和 data/smart_templates/。"
            )
            return

        self.use_camera = False
        self.is_video_file = False
        self.static_frame = None

        if self.cap is None:
            self.cap = self._open_camera()
            self.use_camera = self.cap is not None

        if not self.use_camera:
            initial_dir = str(DATA_DIR / "samples")
            if not os.path.isdir(initial_dir):
                initial_dir = str(DATA_DIR)
            file_path = filedialog.askopenfilename(
                title="选择要检测的图片",
                initialdir=initial_dir,
                filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
            )
            if not file_path:
                return
            try:
                self.static_frame = imread_unicode(file_path)
            except Exception:
                self.static_frame = None
            if self.static_frame is None:
                messagebox.showerror("错误", f"无法读取图片：{file_path}")
                return

        self.running = True
        self.should_speak = True

        # 清掉上一轮残留的框，避免新一轮开头闪到旧画面
        self._last_boxes = []
        self._last_annotated = None
        with self._frame_lock:
            self._latest_frame = None

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        if self.use_camera:
            self.stats_label.config(text="正在扫描（摄像头实时模式）...", fg="#00ff88")
        else:
            self.stats_label.config(text="扫描本地图片中...", fg="#ffcc00")

        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        if self.use_camera:          # 摄像头实时模式才需要主线程 UI 泵；静态图一次性显示不用
            self._start_ui_pump()

    def open_video(self):
        """选一个视频文件，按原生帧率播放并实时检测（复用摄像头双线程逻辑）。

        与摄像头的区别：① 视频读帧不会自带节奏，要按视频 fps 主动限速，否则一闪而过；
        ② 放到结尾要自动停。这两点在 _camera_loop 里按 is_video_file 处理。
        """
        if self.running:
            self.stop_detection()
        if self.detector_engine is None:
            messagebox.showwarning(
                "检测器未就绪",
                f"检测器未就绪，无法开始检测。\n\n详细错误：\n{self.model_error or '未知'}")
            return
        if not self._valid_tools():
            messagebox.showwarning("警告", "没有任何可用工具，无法检测。")
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

        self.running = True
        self.should_speak = True
        self._last_boxes = []
        self._last_annotated = None
        with self._frame_lock:
            self._latest_frame = None

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.stats_label.config(
            text=f"正在检测视频（{os.path.basename(file_path)}）...", fg="#00ff88")

        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        self._start_ui_pump()

    def stop_detection(self):
        self.running = False
        self.should_speak = False
        self._infer_running = False   # 停掉后台推理线程

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.start_btn.config(state="normal", text="▶ 立即检测")
        self.stop_btn.config(state="disabled")
        self.refresh_visible_list()
        self.stats_label.config(text="已停止", fg="#ff6b6b")

    def reset_display(self):
        """清空显示 = 停止当前检测 + 清空 last_seen + 清空画面。

        必须先停检测，否则静态/摄像头线程会立刻把 last_seen 填回去，看起来"没清掉"。
        """
        if self.running:
            self.stop_detection()

        self.last_seen = {}
        self.last_counts = {}
        self._last_boxes = []
        self._last_annotated = None
        try:
            self.refresh_visible_list()
            self.video_label.configure(image="", text="点击立即检测")
            self.video_label.imgtk = None
            self.stats_label.config(text="已清空，待扫描", fg="#ffcc00")
        except Exception as e:
            print(f"清空显示出错: {e}")

    # ---------------- 检测循环 ----------------

    def _show_frame(self, frame_bgr):
        """把 BGR 帧渲染到视频区"""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((900, 650))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

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
        self._last_speak = time.time()   # 首次播报推迟 SPEAK_INTERVAL，别一打开就念一堆
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
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_file:
                    break          # 视频放完 → 退出循环，自动停止
                time.sleep(0.05)   # 摄像头偶发读帧失败 → 重试
                continue

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

        # 列表刷新：识别集合变化就刷（节流 1s），否则每 UI_REFRESH_INTERVAL 兜底
        recog_now = frozenset(n for n, t in list(self.last_seen.items())
                              if now - t < VISIBILITY_WINDOW)
        if ((recog_now != self._last_recog and now - self._last_ui_refresh >= 1.0)
                or now - self._last_ui_refresh >= UI_REFRESH_INTERVAL):
            self.refresh_visible_list()
            self._last_ui_refresh = now
            self._last_recog = recog_now

        # 语音播报（report 非阻塞，瞬间返回）
        if self.should_speak and now - self._last_speak >= SPEAK_INTERVAL:
            self.speak_missing()
            self._last_speak = now

        self.root.after(15, self._ui_pump)   # ~每 15ms 一次，跟随主循环重绘

    def _on_video_finished(self):
        """视频播放结束后在主线程里复位 UI（手动停止不走这里）。"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state="normal", text="▶ 立即检测")
        self.stop_btn.config(state="disabled")
        self.refresh_visible_list()
        self.stats_label.config(text="视频检测完毕", fg="#00ff88")

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
        self._show_frame(display)
        recognized_count = self.refresh_visible_list()
        if self.should_speak:
            self.speak_missing()

        self.running = False
        try:
            self.start_btn.config(state="normal", text="▶ 立即检测")
            self.stop_btn.config(state="disabled")
            total = len(self._valid_tools())
            # 底部数字必须和上方列表一致——用 refresh 返回的 recognized_count，
            # 而不是本帧 len(detected)；连续多次点检测时窗口里旧条目仍算
            self.stats_label.config(
                text=f"检测完成：识别 {recognized_count}/{total}",
                fg="#00ff88")
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
