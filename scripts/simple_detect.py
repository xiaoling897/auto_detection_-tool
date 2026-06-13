"""简化版检测器（旧 ORB 算法，独立运行用于对比）"""
import cv2
import json
import os
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import win32com.client as wincl
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent / "data")


class SimpleToolDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("简单工具检测器")
        self.root.geometry("1400x900")
        self.root.configure(bg="#0a0a1a")

        # 初始化变量
        self.cap = None
        self.running = False
        self.should_speak = False
        self.speaker = wincl.Dispatch("SAPI.SpVoice")
        self.templates = []
        self.load_templates()

        self.setup_ui()

    def load_templates(self):
        # 加载模板图片
        template_dir = "samples"
        if not os.path.exists(template_dir):
            print(f"模板目录 {template_dir} 不存在")
            return

        # 工具名称映射（9 个，顺序与 data/yolo/class_map.json 一致）
        tool_names = [
            "数字万用表", "激光测距仪", "绝缘钢丝钳", "磁力线坠",
            "网线测线仪", "卷尺", "绝缘电阻测试仪", "胎压测试仪", "电工胶带"
        ]

        img_files = sorted([f for f in os.listdir(template_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        for i, img_file in enumerate(img_files[:9]):  # 只取前9个
            img_path = os.path.join(template_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                name = tool_names[i] if i < len(tool_names) else f"工具{i+1}"
                self.templates.append({
                    "name": name,
                    "image": img,
                    "path": img_path
                })
                print(f"加载模板: {name}")

        print(f"共加载 {len(self.templates)} 个模板")

    def detect_tools(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = set()

        for tool in self.templates:
            try:
                result = cv2.matchTemplate(gray, tool["image"], cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                if max_val &gt; 0.6:  # 匹配阈值
                    detected.add(tool["name"])
            except:
                pass

        return detected

    def setup_ui(self):
        # 视频显示区域
        self.video_label = tk.Label(self.root, bg="#0a0a1a", text="点击开始检测")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 控制按钮
        btn_frame = tk.Frame(self.root, bg="#0a0a1a")
        btn_frame.pack(fill=tk.X, padx=20, pady=10)

        self.start_btn = tk.Button(btn_frame, text="开始检测", command=self.start_detection,
                                   font=("Arial", 14), bg="#4CAF50", fg="white", padx=30, pady=10)
        self.start_btn.pack(side=tk.LEFT, padx=10)

        self.stop_btn = tk.Button(btn_frame, text="停止检测", command=self.stop_detection,
                                  font=("Arial", 14), bg="#f44336", fg="white", padx=30, pady=10, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=10)

        # 状态显示
        self.status_label = tk.Label(self.root, text="系统就绪", font=("Arial", 16), bg="#0a0a1a", fg="#00d4ff")
        self.status_label.pack(pady=10)

    def start_detection(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            self.status_label.config(text="无法打开摄像头", fg="red")
            return

        self.running = True
        self.should_speak = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_label.config(text="正在检测...", fg="#4CAF50")

        threading.Thread(target=self.detection_loop, daemon=True).start()

    def stop_detection(self):
        self.running = False
        self.should_speak = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="已停止", fg="#f44336")

    def detection_loop(self):
        last_speak = 0

        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                break

            detected = self.detect_tools(frame)

            # 绘制信息
            h, w = frame.shape[:2]
            cv2.putText(frame, f"检测到: {len(detected)}/{len(self.templates)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            y_pos = 80
            for tool in self.templates:
                status = "✓" if tool["name"] in detected else "✗"
                color = (0, 255, 0) if tool["name"] in detected else (0, 0, 255)
                cv2.putText(frame, f"{status} {tool['name']}", (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                y_pos += 30

            # 显示画面
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img.thumbnail((1200, 800))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

            # 播报
            current_time = time.time()
            if self.should_speak and current_time - last_speak &gt; 8:
                last_speak = current_time
                missing = [t["name"] for t in self.templates if t["name"] not in detected]

                if not missing:
                    self.speak("工具齐全")
                else:
                    for tool in missing:
                        if not self.should_speak:
                            break
                        self.speak(f"{tool}缺失")
                        time.sleep(0.5)
                        self.speak(f"{tool}缺失")
                        time.sleep(0.5)
                        self.speak(f"{tool}缺失")
                        time.sleep(0.5)

            time.sleep(0.03)

    def speak(self, text):
        try:
            if self.should_speak:
                self.speaker.Speak(text)
        except:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleToolDetector(root)
    root.mainloop()
