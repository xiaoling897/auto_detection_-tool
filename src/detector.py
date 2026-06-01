"""SIFT 特征匹配 + Lowe's ratio test + RANSAC 几何校验。

检测核心逻辑。给定一张场景图和一组模板（含 kp/des），返回检测到的工具集合
以及每个工具的 RANSAC 内点数。
"""
import cv2
import numpy as np

INLIER_THRESHOLD = 10
RATIO = 0.75
# 弱匹配通道：内点在 [WEAK_INLIER_MIN, INLIER_THRESHOLD) 之间、且空间高度聚拢（紧凑）时，
# 也算识别。用于捞回无纹理工具（如绝缘钢丝钳）那种"真但弱"的匹配——它内点虽少但紧紧
# 抱在物体上；而散落乱解（如磁力线坠在多工具图里横扫全图）不紧凑，会被挡掉。
# 仅在多工具场景启用（单工具特写走严格阈值，避免放松）。
WEAK_INLIER_MIN = 5
COMPACT_FRAC = 0.3   # 紧凑判定：内点框宽/高均需 < 画面对应边的此比例
# 匹配区域最小散布面积（内点凸包面积，像素²）：RANSAC 有时会接受"退化单应"——
# 把模板所有点塌到一个位置或一条线上，内点数虚高但几何无意义（假匹配）。
# 真实物体匹配的内点会铺满物体轮廓，凸包面积远大于此；实测真匹配最小也 >600，
# 故取 300 既能挡掉塌缩假匹配，又不误伤任何真匹配。
MIN_INLIER_AREA = 300
# 自适应阈值参数：当画面里出现"压倒性匹配"时（≥ STRONG_PEAK），
# 把阈值提到 max × DOMINANT_RATIO，挡掉低质量的偶然匹配
STRONG_PEAK = 300
DOMINANT_RATIO = 0.3

# ---- 颜色旁路：绝缘钢丝钳 ----
# 钢丝钳手柄是红+黄双色，这是它独有的、且不随视角/角度变化的特征；而 SIFT 只看灰度，
# 把颜色信息全扔了，导致它在任何场景都只有 5~7 内点、识别不出。用"红黄相邻块"面积补这个盲区。
# 实测：有钳子的图占比 3.8~9.9‰，无钳子的图 ≤0.61‰，取 1.2‰ 分离干净、误报极低。
PLIERS_NAME = "绝缘钢丝钳"
PLIERS_COLOR_THRESHOLD = 1.2   # 红黄相邻最大连通块占画面比例（千分比）


def red_yellow_pliers_score(frame_bgr):
    """画面里"红、黄两色相邻"的最大连通块占整图的比例（千分比）。

    钢丝钳红黄手柄相挨，膨胀后红区与黄区会重叠；纯红或纯黄的工具（万用表、网线测线仪等）
    不会产生红黄重叠，所以这个分数对钢丝钳高度特异。颜色检测天然不随角度/尺度变化。
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    red = (cv2.inRange(hsv, (0, 90, 70), (10, 255, 255))
           | cv2.inRange(hsv, (160, 90, 70), (179, 255, 255)))
    yellow = cv2.inRange(hsv, (18, 90, 90), (36, 255, 255))
    # 膨胀核随分辨率缩放（让"相邻"判定与画面大小无关）
    ksz = max(11, int(min(h, w) * 0.02))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    inter = cv2.bitwise_and(cv2.dilate(red, k), cv2.dilate(yellow, k))
    inter = cv2.morphologyEx(inter, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(inter)
    big = int(max(stats[1:, cv2.CC_STAT_AREA])) if n > 1 else 0
    return big / (h * w) * 1000.0


class ToolDetector:
    def __init__(self, tools, detector=None):
        """
        Args:
            tools: load_tools() 返回的列表，每项需含 kp / des
            detector: cv2 特征检测器（默认 SIFT，需与 loader 使用的一致）
        """
        self.tools = tools
        self.detector = detector or cv2.SIFT_create(nfeatures=8000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.inlier_threshold = INLIER_THRESHOLD
        self.ratio = RATIO

    def detect(self, frame_bgr):
        """对一帧 BGR 图像执行检测。

        Returns:
            tuple: (detected_set, display_frame, inlier_counts)
              - detected_set: 检测到的工具名集合
              - display_frame: 标注后的 BGR 画面（带"Tools: N/M"提示）
              - inlier_counts: {tool_name: RANSAC 内点数}
        """
        display = frame_bgr.copy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.detector.detectAndCompute(gray, None)

        detected = set()
        counts = {}

        if des_frame is None or len(kp_frame) < 10:
            cv2.putText(display, "No features in frame", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return detected, display, counts

        compacts = {}
        for tool in self.tools:
            name = tool["name"]
            # 多模板：每张模板都匹配一次，取内点最多的那次（最佳视角命中）
            templates = tool.get("templates")
            if not templates and tool.get("des") is not None:
                templates = [{"kp": tool["kp"], "des": tool["des"]}]
            best_n, best_compact = 0, False
            for tpl in (templates or []):
                kp_t, des_t = tpl.get("kp"), tpl.get("des")
                if des_t is None or kp_t is None or len(kp_t) < 4:
                    continue
                n, compact = self._match_one(
                    kp_t, des_t, kp_frame, des_frame, frame_bgr.shape)
                if n > best_n:
                    best_n, best_compact = n, compact
            counts[name] = best_n
            compacts[name] = best_compact

        # 自适应阈值：若画面里有压倒性匹配（说明是单工具特写），
        # 把阈值抬到 max×30%，挡掉偶然匹配；否则用固定阈值（多工具场景）。
        # 弱匹配通道也只在非"压倒性"场景（多工具）开放，单工具特写保持严格。
        max_inliers = max(counts.values()) if counts else 0
        if max_inliers >= STRONG_PEAK:
            threshold = max(self.inlier_threshold, int(DOMINANT_RATIO * max_inliers))
            allow_weak = False
        else:
            threshold = self.inlier_threshold
            allow_weak = True

        for name, inliers in counts.items():
            if inliers >= threshold:
                detected.add(name)
            elif (allow_weak and inliers >= WEAK_INLIER_MIN
                  and compacts.get(name)):
                # 弱但紧凑：无纹理工具的"真但弱"匹配，捞回
                detected.add(name)

        # 颜色旁路：绝缘钢丝钳靠红黄手柄识别，补 SIFT 看不到颜色的盲区（跨视角稳定）
        if any(t["name"] == PLIERS_NAME for t in self.tools):
            score = red_yellow_pliers_score(frame_bgr)
            if score >= PLIERS_COLOR_THRESHOLD:
                detected.add(PLIERS_NAME)
                # 显示用：颜色分数换算成一个和内点数同量级的正整数，列表里显示为已识别
                counts[PLIERS_NAME] = max(counts.get(PLIERS_NAME, 0),
                                          int(score * 10))

        total = sum(1 for t in self.tools if t.get("des") is not None)
        cv2.putText(display, f"Tools: {len(detected)}/{total}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return detected, display, counts

    def _match_one(self, kp_t, des_t, kp_frame, des_frame, frame_shape=None):
        """模板与场景的特征匹配。

        Returns:
            tuple: (内点数, 是否空间紧凑)
              - 内点数：退化单应（内点塌成点/线）判 0
              - 是否紧凑：内点框宽高均 < 画面对应边 COMPACT_FRAC，说明匹配聚拢在一个
                小区域（真落在某个工具上），用于弱匹配通道判定；缺画面尺寸时为 False
        """
        try:
            knn = self.matcher.knnMatch(des_t, des_frame, k=2)
        except cv2.error:
            return 0, False

        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)

        if len(good) < 8:
            return 0, False

        src_pts = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None:
            return 0, False

        inlier_mask = mask.ravel().astype(bool)
        n_inliers = int(inlier_mask.sum())
        if n_inliers < 4:
            return n_inliers, False

        # 几何合理性校验：内点在场景里的凸包面积太小 = 退化单应（塌成点/线），判为假匹配
        inl_pts = dst_pts.reshape(-1, 2)[inlier_mask]
        hull_area = cv2.contourArea(cv2.convexHull(inl_pts.astype(np.float32)))
        if hull_area < MIN_INLIER_AREA:
            return 0, False

        # 紧凑度：内点框相对画面够小 = 聚拢在一个工具上（真匹配特征）
        compact = False
        if frame_shape is not None:
            img_h, img_w = frame_shape[:2]
            x1, y1 = inl_pts.min(axis=0)
            x2, y2 = inl_pts.max(axis=0)
            compact = ((x2 - x1) < COMPACT_FRAC * img_w
                       and (y2 - y1) < COMPACT_FRAC * img_h)

        return n_inliers, compact
