"""YOLOv8 工具检测器。

与 detector.ToolDetector 同接口（detect 返回 (detected_set, display_frame, counts)），
可直接被 gui.py 替换调用，不影响上层 UX 逻辑。

模型文件约定放在 data/yolo/best.pt；英文→中文类名映射在 data/yolo/class_map.json。
"""
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CONFIDENCE_THRESHOLD = 0.25   # YOLO 输出置信度阈值（降低让弱检测也通过）
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

    def detect(self, frame_bgr):
        """对一帧 BGR 图像跑 YOLO 检测。

        Returns:
            tuple: (detected_set, display_frame, confidences)
              - detected_set: 检测到的工具中文名集合
              - display_frame: 标注后的 BGR 画面（带框 + 中文标签）
              - confidences: {tool_name: 置信度 0-100 整数}——同 SIFT 检测器输出格式，
                             方便 GUI 不改地复用（之前显示"内点数"的位置现在显示置信度%）
        """
        results = self.model(
            frame_bgr,
            conf=self.confidence_threshold,
            iou=IOU_THRESHOLD,
            verbose=False,
        )
        result = results[0]

        display = frame_bgr.copy()
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
        total = len(self.class_map) if self.class_map else len(self.model.names)
        cv2.putText(
            display, f"Tools: {len(detected)}/{total}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        return detected, display, confidences
