# 智能工具检测系统 - 项目笔记

## 这是什么

基于 **YOLOv8 深度学习** 的工具识别 GUI 应用，识别 11 种电工工具（钢直尺、螺丝刀、绝缘钢丝钳、数字万用表、绝缘电阻测试仪、网线测线仪、激光测距仪、数显倾角仪、电工胶带、磁力线坠、胎压测试仪）。

**之前版本基于 SIFT 多尺度特征匹配 + RANSAC**——在多工具固定监控场景下识别率只能到 7/11（4 个表面无纹理的工具识别不了），所以迁移到了 YOLO。SIFT 相关代码 [src/detector.py](src/detector.py) 和 [src/loader.py:load_tools](src/loader.py) 还保留着作为参考/回退方案。

## 运行入口

```bash
python run.py
```
读取 [data/smart_tools.json](data/smart_tools.json) 加载模板，启动 Tkinter GUI。Windows 专用（语音用 SAPI）。

## 目录结构

```
run.py                  # 入口
src/                    # 核心源码（包，被 run.py 引用）
  gui.py                # Tkinter UI + 实时扫描主循环（3s 滑动窗口）
  yolo_detector.py      # ⭐ YOLOv8 检测核心（当前生效）
  detector.py           # SIFT + RANSAC 检测核心（旧版，保留作参考/回退）
  loader.py             # smart_tools.json 解析（load_tool_names YOLO 用，load_tools SIFT 用）
  voice.py              # Windows SAPI 语音
  utils.py              # 中文路径读图 + DATA_DIR 路径解析
data/                   # 所有数据资源
  smart_tools.json      # 工具名清单（YOLO 模式只读 name 字段）
  yolo/
    best.pt             # ⭐ YOLO 训练产物，用户自己训练后放这里
    class_map.json      # 英文类名 → 中文显示名映射
  smart_templates/      # 旧 SIFT 模板图（YOLO 不用，保留兼容）
  samples/              # 测试用图片
scripts/                # 一次性脚本
tests/                  # 测试
```

## 检测算法（关键，别改坏）

### YOLOv8 检测流程（当前生效）

1. 加载 `data/yolo/best.pt`（用户自己训练）
2. 每帧调用 `model(frame, conf=0.5, iou=0.45)`——置信度阈值 0.5，NMS 阈值 0.45
3. 每个 box 的 class_id → `model.names[class_id]` 得到英文类名 → 查 `data/yolo/class_map.json` 转中文名
4. 同一中文名出现多个框时取**最高置信度**（防止同一工具误算两次）
5. 输出 `(detected_set, display_frame, confidences)`——和旧 SIFT detector 同接口，GUI 不用动

### 关键参数（[src/yolo_detector.py](src/yolo_detector.py) 顶部）

- `CONFIDENCE_THRESHOLD = 0.5`：越低越宽松（更易误检），越高越严（更易漏检）
- `IOU_THRESHOLD = 0.45`：NMS 去重阈值，一般不动

### SIFT 算法演进史（旧版本，仅供回顾）

| 阶段 | 算法 | 单工具图 | 多工具图 |
|------|------|---------:|--------:|
| v1 | ORB 单尺度 + 绝对阈值 | 11/11 但每张误识别 6-7 个 | 0/11 |
| v2 | + RANSAC | 11/11 零误识别 | 0/11 |
| v3 | + SIFT 多尺度模板 | 退化：每张又误识别 4-10 个 | 7/11 |
| v4 | + 自适应阈值 | 11/11 零误识别 | 7/11 |
| **v5（当前）** | **YOLOv8** | **11/11**（数据训练得好的话） | **11/11** |

**为什么放弃 SIFT**：那 4 个识别不了的工具（磁力线坠、绝缘钢丝钳、螺丝刀、钢直尺）表面光滑无纹理，SIFT 特征点提取不出来——这是特征匹配类算法的物理上限。YOLO 学的是物体整体外观语义，不依赖纹理，所以能突破这个上限。

## 实时扫描 UX 模型（核心）

**滑动窗口 + 全量展示**：右侧清单**始终列出全部 11 个工具**，但**识别到的（✓ 绿）排上面，未识别的（✗ 红，名字灰）排下面**。识别判定依据"最近 `VISIBILITY_WINDOW = 3.0s` 内 `last_seen` 有更新"——工具拿走 ≥3s 会从上半区掉到下半区，放回镜头前下一次检测命中就升上去。

- 检测线程仍每 `DETECT_INTERVAL = 0.4s` 跑一次（实时更新 `last_seen` 时间戳）
- UI 列表每 `UI_REFRESH_INTERVAL = 3.0s` 刷新一次（不实时刷新，避免行频繁 pack/forget 抖动）
- 每次 refresh 都先 forget 所有行再按"识别优先 + 配置顺序"重新 pack——`recognized + unrecognized` 两组拼接，组内保持工具配置顺序，所以视觉上稳定不抖
- 状态指示：✓ 绿（识别）/ ✗ 红（未识别）；内点数 ≥10 绿、5-9 黄、<5 灰——未识别行整体灰显
- 启动时立即调一次 `refresh_visible_list`，让 11 个工具初始全部以 ✗ 显示

| 按钮 | 行为 |
|------|------|
| **▶ 立即检测** | 每次点击都独立：有摄像头进入连续扫描循环；无摄像头弹文件框选一张 |
| **⏹ 停止检测** | 手动停止扫描，列表停留在停止瞬间的状态 |
| **🔄 清空显示** | 立即清空 `last_seen` + 清空列表（摄像头若仍在跑，下一帧识别到会自动重新出现） |

语音播报间隔 8 秒，逐个念当前"未在 3s 窗口内"的工具。摄像头**不再**自动停止（之前的"累计 11/11 自动停"逻辑已删除）。

## 关键陷阱

- **YOLO 模型不在仓库里**：`data/yolo/best.pt` 需要用户自己训练后放进去（仓库里只有 `class_map.json` 模板）。GUI 启动时检测不到模型不会崩，但点"立即检测"会弹提示。
- **class_map.json 的 key 必须和训练时的英文类名一字不差**：训练时在 Roboflow / data.yaml 写的 `ruler/screwdriver/...` 这些键名，必须和 `data/yolo/class_map.json` 的 key 完全对应，否则会显示英文名而不是中文名。
- **首次 `pip install ultralytics`**：会拉 PyTorch（约 500MB-1GB），CPU 版的不需要 CUDA，普通电脑能装。
- **ultralytics import 慢**（2-5s）：[src/yolo_detector.py](src/yolo_detector.py) 把 `from ultralytics import YOLO` 放在 `__init__` 里延迟 import，避免没用 YOLO 时也付出这个代价。
- **中文标签画框**：cv2.putText 不支持中文，[src/yolo_detector.py](src/yolo_detector.py) 用 PIL + 微软雅黑（`C:/Windows/Fonts/msyh.ttc`）画中文标签，cv2 只画矩形框——单帧多框时合并一次 cv2↔PIL 转换提速。
- **中文路径**：Windows + 中文路径下 `cv2.imread` 会**静默失败**，必须用 [src.utils.imread_unicode](src/utils.py)（`np.frombuffer` + `cv2.imdecode`）。当前 GUI 文件框选静态图时用的就是这个。
- **摄像头打开慢**：Windows DSHOW 后端，无摄像头时 `VideoCapture(0, CAP_DSHOW)` 会**卡 7-9 秒**才返回失败，这是 OpenCV 行为。GUI 每次点"立即检测"都会重新探测，没办法绕过。
- **`smart_tools.json` 的 `features` 字段是历史遗留**（旧 ORB descriptors，4.8MB）。YOLO 模式只读 `name` 字段（[load_tool_names](src/loader.py)）。保留 features 是为旧脚本兼容。
- **检测线程更新 Tkinter UI**：现有代码从 detection 线程直接调 `.config()` / `pack()`，Tkinter 严格上非线程安全但实际能用——别加锁也别引入信号机制，保持现状。

## 常见操作

```bash
# 验证算法（批量测试）
python tests/test_image_detect.py                       # 跑 data/samples 全部
python tests/test_image_detect.py data/samples/all_tools.jpg

# 重建 smart_tools.json（加新模板后）
python scripts/extract_features.py

# 打包 EXE
python scripts/build_exe.py
```
