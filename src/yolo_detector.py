"""YOLOv8 工具检测器。

与 detector.ToolDetector 同接口（detect 返回 (detected_set, display_frame, counts)），
可直接被 gui.py 替换调用，不影响上层 UX 逻辑。

模型文件约定放在 data/yolo/best.pt；英文→中文类名映射在 data/yolo/class_map.json。
"""
import json
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CONFIDENCE_THRESHOLD = 0.55   # YOLO 置信度阈值（偏准确）。0.70 太严会漏检远处/小目标；
                              # 0.40 太松会带进 47~50% 的垃圾框/误检。0.55 折中偏准：
                              # 砍掉大部分低置信度误检，代价是极弱的真目标(如某些角度的钢丝钳)可能漏。
                              # 注意：高置信度"张冠李戴"(如红色万用表被认成绝缘电阻测试仪)是模型混淆，
                              # 调阈值无效，只能补训练照片重训。误检多就往上调、漏检多就往下调。
IOU_THRESHOLD = 0.45          # NMS 阈值

# Windows 中文字体路径（cv2.putText 不支持中文，得走 PIL）
_CN_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",    # 黑体
    "C:/Windows/Fonts/simsun.ttc",    # 宋体
]


class YoloToolDetector:
    def __init__(self, model_path, class_map_path=None, conf=None):
        """
        Args:
            model_path: 训练好的 .pt 模型文件路径
            class_map_path: JSON 文件路径，{英文类名: 中文显示名}
                            找不到映射时直接用英文名
            conf: 置信度阈值（默认 0.5）
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"YOLO 模型文件不存在: {model_path}\n"
                f"请先训练模型，然后把 best.pt 放到这个路径。"
            )

        # 延迟 import：ultralytics 启动慢（要加载 PyTorch），
        # 只在真正构造检测器时才付出代价
        from ultralytics import YOLO

        # 限制 torch 推理线程数：默认 torch 会吃满所有 CPU 核，导致 GUI 的采集/显示线程
        # 抢不到 CPU，摄像头画面卡顿。留 2 个核给界面与摄像头采集（至少留 1 个），
        # 纯 CPU 机器上视频明显更顺——代价是单次推理略慢，但检测仍是实时几帧/秒。
        try:
            import torch
            n_cpu = os.cpu_count() or 4
            torch.set_num_threads(max(1, n_cpu - 2))
            print(f"🧵 torch 推理线程数限制为 {max(1, n_cpu - 2)}（共 {n_cpu} 核，留核给界面）")
        except Exception as e:
            print(f"设置 torch 线程数失败（忽略）: {e}")

        self.model = YOLO(model_path)
        self.confidence_threshold = conf if conf is not None else CONFIDENCE_THRESHOLD

        self.class_map = {}
        if class_map_path and os.path.isfile(class_map_path):
            try:
                with open(class_map_path, "r", encoding="utf-8") as f:
                    self.class_map = json.load(f)
            except Exception as e:
                print(f"读取 class_map 失败，将直接使用模型类名: {e}")

        self._font = self._load_font(20)
        self._font_small = self._load_font(16)
        self.last_boxes = []   # 最近一次 infer 的框，供实时显示线程复用

        # 预热：YOLO 第一次推理要做一堆延迟初始化（冷启动好几秒）。在这里先空跑一帧，
        # 把这笔开销提前到启动时付掉——否则"打开视频/摄像头"后第一次检测会卡好几秒才出框
        # （旧单线程版是靠"第一帧卡住等推理"掩盖了这点；新双线程版视频已在流畅播放，
        #  冷启动那几秒就暴露成"放了半天不出框"）。
        try:
            t0 = time.time()
            self.model(np.zeros((480, 640, 3), dtype=np.uint8),
                       conf=self.confidence_threshold, iou=IOU_THRESHOLD, verbose=False)
            print(f"🔥 YOLO 预热完成（{(time.time() - t0) * 1000:.0f} ms），首次检测即时出框")
        except Exception as e:
            print(f"YOLO 预热失败（忽略，不影响功能）: {e}")
        # 中文标签贴图缓存：label 字符串 -> 预渲染好的小图（BGR）。
        # 实时显示线程每帧画框，若每帧都做"整幅 cv2<->PIL 转换"画中文，720p 上要十几毫秒，
        # 会拖慢视频。改成：每个标签只用 PIL 渲染一次成小贴图缓存起来，之后每帧只做一次
        # numpy 切片贴图（几乎零开销）—— 这是双线程下保证视频顺的关键。
        self._label_cache = {}

    @staticmethod
    def _load_font(size):
        for path in _CN_FONT_CANDIDATES:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _to_cn(self, model_name):
        """模型英文类名 → 中文显示名（找不到映射就用原名）"""
        return self.class_map.get(model_name, model_name)

    def infer(self, frame_bgr):
        """只跑推理，不画图。给实时检测线程用——视频线程不必等画图。

        Returns:
            tuple: (detected_set, confidences, boxes)
              - detected_set: 检测到的工具中文名集合
              - confidences: {tool_name: 置信度 0-100 整数}
              - boxes: [(x1, y1, x2, y2, cn_name, conf_pct), ...] 供 render 复用
        """
        # 用默认推理尺寸（640）：416 虽快但小目标/远处工具像素不足会漏检。
        # 速度问题靠"限制 torch 线程 + 检测线程冷却 + 双线程解耦"解决，不靠牺牲分辨率。
        results = self.model(
            frame_bgr,
            conf=self.confidence_threshold,
            iou=IOU_THRESHOLD,
            verbose=False,
        )
        result = results[0]

        detected = set()
        confidences = {}
        boxes_to_draw = []

        if result.boxes is not None and len(result.boxes) > 0:
            model_names = self.model.names
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)

                if isinstance(model_names, dict):
                    model_name = model_names.get(cls_id)
                else:
                    model_name = model_names[cls_id] if 0 <= cls_id < len(model_names) else None
                if model_name is None:
                    continue

                cn_name = self._to_cn(model_name)
                detected.add(cn_name)
                confidences[cn_name] = max(confidences.get(cn_name, 0), int(conf * 100))

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes_to_draw.append((x1, y1, x2, y2, cn_name, int(conf * 100)))

        self.last_boxes = boxes_to_draw
        return detected, confidences, boxes_to_draw

    def render(self, frame_bgr, boxes_to_draw):
        """把 boxes 画到帧上，返回标注后的 BGR 帧。

        实时显示线程每帧调用——可复用最近一次 infer 的 boxes，使框在两次检测之间
        持续显示在实时画面上（固定监控场景里目标基本不动，框位置稳定）。
        """
        display = frame_bgr.copy()

        # 矩形框用 cv2 画（快）
        for x1, y1, x2, y2, _, _ in boxes_to_draw:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 中文标签用 PIL 画（一次性转换，避免每个标签都做 cv2<->PIL 转换）
        if boxes_to_draw:
            img_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            drawer = ImageDraw.Draw(img_pil)
            for x1, y1, _, _, cn_name, conf_pct in boxes_to_draw:
                label = f"{cn_name} {conf_pct}%"
                y_text = max(y1 - 24, 0)
                drawer.text((x1, y_text), label, font=self._font, fill=(0, 255, 0))
            display = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 顶部统计（英文，cv2 即可）
        detected_count = len({b[4] for b in boxes_to_draw})
        total = len(self.class_map) if self.class_map else len(self.model.names)
        cv2.putText(
            display, f"Tools: {detected_count}/{total}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        return display

    def _label_sprite(self, text):
        """把一段中文标签预渲染成小贴图（BGR，黑底绿字），按文本缓存。

        只有遇到没见过的标签字符串才会真正走一次 PIL 渲染；之后命中缓存直接返回，
        所以实时显示线程每帧叠标签几乎不花 CPU。
        """
        cached = self._label_cache.get(text)
        if cached is not None:
            return cached
        dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bbox = dummy.textbbox((0, 0), text, font=self._font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 3
        img = Image.new("RGB", (tw + 2 * pad, th + 2 * pad), (0, 0, 0))
        ImageDraw.Draw(img).text(
            (pad - bbox[0], pad - bbox[1]), text, font=self._font, fill=(0, 255, 0))
        spr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        self._label_cache[text] = spr
        return spr

    def render_live(self, frame_bgr, boxes_to_draw):
        """实时显示专用：cv2 画矩形框 + 贴缓存好的中文标签，每帧开销极小。

        与 render() 的区别：render() 每帧都做整幅 cv2<->PIL 转换（慢），适合静态图；
        render_live() 用预渲染的标签贴图做 numpy 切片叠加，适合摄像头实时显示线程逐帧调用。
        直接在传入的 frame_bgr 上就地绘制（调用方自行决定是否先 copy）。
        """
        h, w = frame_bgr.shape[:2]
        for x1, y1, x2, y2, cn_name, conf_pct in boxes_to_draw:
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            spr = self._label_sprite(f"{cn_name} {conf_pct}%")
            sh, sw = spr.shape[:2]
            if sw > w or sh > h:
                continue
            y = max(0, y1 - sh)
            x = max(0, min(x1, w - sw))
            frame_bgr[y:y + sh, x:x + sw] = spr

        detected_count = len({b[4] for b in boxes_to_draw})
        total = len(self.class_map) if self.class_map else len(self.model.names)
        cv2.putText(
            frame_bgr, f"Tools: {detected_count}/{total}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        return frame_bgr

    def detect(self, frame_bgr):
        """对一帧 BGR 图像跑 YOLO 检测（推理 + 画图一步到位，静态图/旧接口用）。

        Returns:
            tuple: (detected_set, display_frame, confidences)——同 SIFT 检测器输出格式。
        """
        detected, confidences, boxes = self.infer(frame_bgr)
        display = self.render(frame_bgr, boxes)
        return detected, display, confidences
