"""Tkinter GUI：实时扫描模式（检测后端自动选择：有 data/yolo/best.pt 用 YOLO，否则 SIFT；3 秒滑动窗口防抖）

工作流：
  - 摄像头模式：持续读帧 → 每 0.4s 跑一次 SIFT 特征匹配 → 记录每个工具的最近识别时间戳
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

DETECT_INTERVAL = 0.4       # 检测节流（CPU 友好）
UI_REFRESH_INTERVAL = 3.0   # 列表 UI 刷新节奏
VISIBILITY_WINDOW = 3.0     # 工具可见性窗口：最近 N 秒内识别到才显示
SPEAK_INTERVAL = 3.0        # 检测出结果后等待 3 秒再播报


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
        self.static_frame = None
        self.running = False
        self.should_speak = False
        self.last_seen = {}      # 工具名 -> 上次识别到的时间戳
        self.last_counts = {}    # 工具名 -> 上一帧 RANSAC 内点数
        self.last_source = {}    # 工具名 -> 上一帧命中来源（"强"/"弱"/"色"）
        self.detect_threshold = 10   # 上一帧有效阈值（自适应，诊断显示用）

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
        self.static_frame = None

        if self.cap is None:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, test_frame = cap.read()
                # 有些 Windows 没有真实摄像头时，VideoCapture 仍会"打开"一个返回
                # 大片均匀灰屏的虚拟设备——首帧方差极小说明不是真画面，也按无摄像头处理，
                # 回退到让用户选图片。
                usable = (ret and test_frame is not None
                          and float(test_frame.std()) > 6.0)
                if usable:
                    self.cap = cap
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.use_camera = True
                else:
                    cap.release()
            else:
                cap.release()

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

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        if self.use_camera:
            self.stats_label.config(text="正在扫描（摄像头实时模式）...", fg="#00ff88")
        else:
            self.stats_label.config(text="扫描本地图片中...", fg="#ffcc00")

        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()

    def stop_detection(self):
        self.running = False
        self.should_speak = False

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

    def _camera_loop(self):
        last_speak = None
        last_detect = 0.0
        last_ui_refresh = 0.0
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            now = time.time()
            display = frame
            if now - last_detect >= DETECT_INTERVAL:
                detected, display, counts = self.detector_engine.detect(frame)
                # 每帧严格检测，命中即更新 last_seen；多帧累计 + 3 秒窗口决定"在场/缺失"。
                # 严格检测在场景里没有持久假阳性，所以工具拿走后 3 秒内会自动掉为缺失。
                for name in detected:
                    self.last_seen[name] = now
                self.last_counts.update(counts)
                # 来源/阈值整帧替换（SIFT 才有；YOLO 无此属性，getattr 兜底）
                self.last_source = dict(getattr(self.detector_engine, "last_source", {}))
                self.detect_threshold = getattr(self.detector_engine, "last_threshold", 0)
                last_detect = now
                # 诊断：每帧把"检测到的工具 + 数值(SIFT内点/YOLO置信%) + 来源"打到控制台
                hits = " ".join(
                    f"{n}={counts.get(n, 0)}{self.last_source.get(n, '')}"
                    for n in sorted(detected))
                print(f"[检测 {len(detected)}个] {hits or '(空)'}")
                if last_speak is None:
                    last_speak = now

            self._show_frame(display)

            # 列表 UI 每 3 秒刷新一次（不实时刷新，避免闪烁）
            if now - last_ui_refresh >= UI_REFRESH_INTERVAL:
                self.refresh_visible_list()
                last_ui_refresh = now

            if (self.should_speak and last_speak is not None
                    and now - last_speak >= SPEAK_INTERVAL):
                self.speak_missing()
                last_speak = now

            time.sleep(0.02)

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
