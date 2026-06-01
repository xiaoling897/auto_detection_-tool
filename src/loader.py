"""加载工具配置和模板特征。

多尺度模板：把每个模板缩放到 (1.0, 0.6, 0.4, 0.25) 后分别提取 SIFT 特征再合并，
这样无论场景里工具是大特写还是远景小目标都能匹配上。
"""
import json
import cv2
import numpy as np

from .utils import imread_unicode, resolve_data_path, DATA_DIR

DEFAULT_SCALES = (1.0, 0.6, 0.4, 0.25)
DEFAULT_NFEATURES = 8000


def multiscale_features(detector, img, scales=DEFAULT_SCALES):
    """对一张模板图在多个尺度下提取特征，合并后返回 (kp_list, des_array)"""
    h, w = img.shape[:2]
    all_kp, all_des = [], []
    for s in scales:
        if s == 1.0:
            scaled = img
        else:
            scaled = cv2.resize(img, (int(w * s), int(h * s)),
                                interpolation=cv2.INTER_AREA)
        kp, des = detector.detectAndCompute(scaled, None)
        if des is None or len(kp) == 0:
            continue
        # 关键点坐标缩放回原模板尺度（RANSAC 几何校验需要）
        for k in kp:
            k.pt = (k.pt[0] / s, k.pt[1] / s)
            k.size = k.size / s
        all_kp.extend(kp)
        all_des.append(des)
    if not all_des:
        return [], None
    return all_kp, np.vstack(all_des)


def load_tools(config_path=None, detector=None, scales=DEFAULT_SCALES, verbose=True):
    """加载工具配置 + 提取所有模板的多尺度特征。

    Returns:
        list[dict]: 每个工具含 name / template_path / kp / des 字段
    """
    if config_path is None:
        config_path = DATA_DIR / "smart_tools.json"
    if detector is None:
        detector = cv2.SIFT_create(nfeatures=DEFAULT_NFEATURES)

    with open(config_path, 'r', encoding='utf-8') as f:
        tools = json.load(f)
    if verbose:
        print(f"已加载 {len(tools)} 个工具配置")

    templates_dir = DATA_DIR / "smart_templates"
    for tool in tools:
        tool["kp"] = None
        tool["des"] = None
        tool["templates"] = []      # 多模板：[{kp, des, path}, ...]，匹配时取最佳
        name = tool.get("name", "")

        # 收集模板图：主模板 template_path + 同名额外视角 <工具名>_*.{jpg,png,...}。
        # 额外视角用于补不同角度/场景（如俯拍监控视角），丢一张同名文件即自动生效。
        paths = []
        primary = resolve_data_path(tool.get("template_path", ""))
        if primary.exists():
            paths.append(primary)
        for pat in (f"{name}_*.jpg", f"{name}_*.jpeg", f"{name}_*.png", f"{name}_*.bmp"):
            paths.extend(sorted(templates_dir.glob(pat)))
        # 去重，保持顺序（主模板在前）
        seen, uniq = set(), []
        for p in paths:
            key = str(p).lower()
            if key not in seen:
                seen.add(key)
                uniq.append(p)

        if not uniq and verbose:
            print(f"  模板缺失: {name} -> {primary}")

        for p in uniq:
            try:
                img = imread_unicode(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                kp, des = multiscale_features(detector, img, scales)
                if des is None or not kp:
                    continue
                tool["templates"].append({"kp": kp, "des": des, "path": str(p)})
            except Exception as e:
                if verbose:
                    print(f"  加载模板失败 {name} ({p.name}): {e}")

        if tool["templates"]:
            # 主模板的 kp/des 暴露在顶层，兼容只看 tool['des'] 的旧代码
            tool["kp"] = tool["templates"][0]["kp"]
            tool["des"] = tool["templates"][0]["des"]
        if verbose and len(tool["templates"]) > 1:
            print(f"  {name}: {len(tool['templates'])} 张模板")

    if verbose:
        valid = sum(1 for t in tools if t.get("des") is not None)
        print(f"特征提取完毕：{valid}/{len(tools)}")
    return tools


def load_tool_names(class_map_path=None):
    """YOLO 模式用：从 class_map.json 拿工具中文名列表。

    class_map.json 是 YOLO 模式下工具集的唯一权威——决定模型识别哪些类。
    顺序就是 JSON 文件里 key 的顺序（Python 3.7+ 保留插入顺序）。

    Returns:
        list[str]: 工具中文名列表，例如 ["数字万用表", "激光测距仪", ...]
    """
    if class_map_path is None:
        class_map_path = DATA_DIR / "yolo" / "class_map.json"
    if not class_map_path.exists():
        return []
    with open(class_map_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return list(mapping.values())
