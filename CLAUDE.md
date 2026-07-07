# 智能工具检测系统 - 项目笔记

## 这是什么

基于 **YOLOv8 深度学习** 的工具识别 GUI 应用，识别 9 种电工工具（数字万用表、激光测距仪、绝缘钢丝钳、磁力线坠、网线测线仪、卷尺、绝缘电阻测试仪、胎压测试仪、电工胶带）。

> 工具清单的唯一权威来源是 [data/yolo/class_map.json](data/yolo/class_map.json)（英文类名→中文名）和 [data/smart_tools.json](data/smart_tools.json)（GUI 读 `name`），两者都是这 9 个、顺序一致。改工具一定要同步这两个文件 + 重新训练。

**之前版本基于 SIFT 多尺度特征匹配 + RANSAC**——在多工具固定监控场景下识别率上不去（好几个表面无纹理的工具识别不了），所以迁移到了 YOLO。SIFT 相关代码 [src/detector.py](src/detector.py) 和 [src/loader.py:load_tools](src/loader.py) 还保留着作为参考/回退方案。

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
data/                   # 数据资源（根目录只放运行时需要的）
  smart_tools.json      # 工具名清单（YOLO 模式只读 name 字段）
  yolo/
    best.pt             # ⭐ YOLO 训练产物，用户自己训练后放这里
    class_map.json      # 英文类名 → 中文显示名映射
  smart_templates/      # 旧 SIFT 模板图（YOLO 不用，保留兼容）
  samples/              # 测试用图片
  training/             # ⭐ 训练相关全在这（整个目录已 gitignore，不进仓库）
    photo/              #   原始拍摄图（sam_autolabel 的输入源）
    new_shots2/         #   摄像头补拍的原图
    autolabel/          #   SAM 自动标注产物（prepare_dataset 的输入）
    label_multi/        #   labelImg 手工标注的多工具合照
    label_new*/ label_full/  # 其它手工标注批次（prepare_dataset 按 label_* 自动合并）
    yolo_dataset/       #   最终 YOLO 训练集（prepare_dataset 切分产物，含 data.yaml）
    runs/               #   train_yolo 的训练日志/权重输出
models/                 # 预训练底座权重（已 gitignore，可重新下载）
  yolo11s.pt            #   训练底座（train_yolo.py）
  FastSAM-s.pt          #   自动标注底座（sam_autolabel.py / sam_boxall.py）
scripts/                # 一次性脚本（数据路径指向 data/training/，权重路径指向 models/）
tests/                  # 测试
```
> 注：`VisionGuard-AI_Detection_System/`（手势/姿态检测，独立 git 仓库）原先嵌在本项目里，已移出到 `d:/VisionGuard-AI_Detection_System`，与本项目无关。

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
| v1 | ORB 单尺度 + 绝对阈值 | 单张全识别但每张误识别 6-7 个 | 全军覆没 |
| v2 | + RANSAC | 单张全识别零误识别 | 全军覆没 |
| v3 | + SIFT 多尺度模板 | 退化：每张又误识别 4-10 个 | 部分识别 |
| v4 | + 自适应阈值 | 单张全识别零误识别 | 部分识别 |
| **v5（当前）** | **YOLOv8** | **全识别**（数据训练得好的话） | **全识别** |

**为什么放弃 SIFT**：那几个识别不了的工具（磁力线坠、绝缘钢丝钳 等）表面光滑无纹理，SIFT 特征点提取不出来——这是特征匹配类算法的物理上限。YOLO 学的是物体整体外观语义，不依赖纹理，所以能突破这个上限。

## 实时扫描 UX 模型（核心）

**滑动窗口 + 全量展示**：右侧清单**始终列出全部 9 个工具**，但**识别到的（✓ 绿）排上面，未识别的（✗ 红，名字灰）排下面**。识别判定依据"最近 `VISIBILITY_WINDOW = 3.0s` 内 `last_seen` 有更新"——工具拿走 ≥3s 会从上半区掉到下半区，放回镜头前下一次检测命中就升上去。

- 检测线程仍每 `DETECT_INTERVAL = 0.4s` 跑一次（实时更新 `last_seen` 时间戳）
- **双线程解耦实时**（[_camera_loop](src/gui.py) + [_inference_worker](src/gui.py)）：
  - **显示线程**（`_camera_loop`）：持续读帧 → 每帧用 `render_live` 轻量叠最近的框 → 显示 → 每 `UI_REFRESH_INTERVAL = 3.0s` 刷列表。永远满帧率，不等推理。
  - **推理线程**（`_inference_worker`）：后台每 `DETECT_INTERVAL = 0.4s` 取最新帧跑 YOLO，只回写 `_last_boxes`/`last_seen`/`last_counts`，**不碰 Tkinter**（线程安全）。
  - 两线程靠 `_frame_lock` + `_latest_frame` 传帧。推理慢只拖慢"框刷新率"，**绝不阻塞视频** —— 这是用重模型（yolo11s 9.4M）也不卡的根因修复（2026-06，详见 [tests/bench_camera_smoothness.py](tests/bench_camera_smoothness.py)，单线程→双线程显示帧率实测提升数倍）。
  - **为什么这次双线程不会"饿死显示"**（之前单线程版踩过的坑）：① `yolo_detector` 里 `torch.set_num_threads(n_cpu-2)` 给显示线程留核；② 显示线程每帧只做 cv2 画框 + **缓存好的中文标签贴图**（`render_live`/`_label_sprite`），绝不每帧整幅 `cv2<->PIL` 转换（那是旧版每帧十几 ms 的卡顿来源）。两条缺一不可。
- **框常驻**：每帧都把最近一次推理的框（YOLO 的 `_last_boxes`）叠到当前实时帧上，不是只在推理那一帧画，所以框跟着实时画面、不闪
- 每次 refresh 都先 forget 所有行再按"识别优先 + 配置顺序"重新 pack——`recognized + unrecognized` 两组拼接，组内保持工具配置顺序，所以视觉上稳定不抖
- 状态指示：✓ 绿（识别）/ ✗ 红（未识别）；内点数 ≥10 绿、5-9 黄、<5 灰——未识别行整体灰显
- 启动时立即调一次 `refresh_visible_list`，让 9 个工具初始全部以 ✗ 显示

| 按钮 | 行为 |
|------|------|
| **▶ 立即检测** | 每次点击都独立：有摄像头进入连续扫描循环；无摄像头弹文件框选一张 |
| **⏹ 停止检测** | 手动停止扫描，列表停留在停止瞬间的状态 |
| **🔄 清空显示** | 立即清空 `last_seen` + 清空列表（摄像头若仍在跑，下一帧识别到会自动重新出现） |

语音播报间隔 8 秒，逐个念当前"未在 3s 窗口内"的工具。摄像头**不再**自动停止（之前的"累计 9/9 自动停"逻辑已删除）。

## 关键陷阱

- **YOLO 模型不在仓库里**：`data/yolo/best.pt` 需要用户自己训练后放进去（仓库里只有 `class_map.json` 模板）。GUI 启动时检测不到模型不会崩，但点"立即检测"会弹提示。
- **class_map.json 的 key 必须和训练时的英文类名一字不差**：训练时在 Roboflow / data.yaml 写的 `ruler/screwdriver/...` 这些键名，必须和 `data/yolo/class_map.json` 的 key 完全对应，否则会显示英文名而不是中文名。
- **首次 `pip install ultralytics`**：会拉 PyTorch（约 500MB-1GB），CPU 版的不需要 CUDA，普通电脑能装。
- **ultralytics import 慢**（2-5s）：[src/yolo_detector.py](src/yolo_detector.py) 把 `from ultralytics import YOLO` 放在 `__init__` 里延迟 import，避免没用 YOLO 时也付出这个代价。
- **中文标签画框**：cv2.putText 不支持中文，[src/yolo_detector.py](src/yolo_detector.py) 用 PIL + 微软雅黑（`C:/Windows/Fonts/msyh.ttc`）画中文标签，cv2 只画矩形框——单帧多框时合并一次 cv2↔PIL 转换提速。
- **中文路径**：Windows + 中文路径下 `cv2.imread` 会**静默失败**，必须用 [src.utils.imread_unicode](src/utils.py)（`np.frombuffer` + `cv2.imdecode`）。当前 GUI 文件框选静态图时用的就是这个。
- **摄像头打开慢 / 换 4K 摄像头打不开**：[_open_camera](src/gui.py) 用「尝试矩阵」`_CAM_ATTEMPTS` 逐项试——`DSHOW`/`MSMF` 双后端 × `720p`/`native` 双分辨率 × 选配 `MJPG`，第一项就是旧 USB 摄像头惯用的「DSHOW+720p+默认格式」(老设备零回退)，后面几项给 4K UVC 直播摄像机(如海康 DS-UVC-U168R)兜底。
  - **分辨率/格式必须在预热读帧之前设**：旧代码先用默认分辨率预热、之后才设 720p，4K 机就拿 3840×2160 大帧预热导致读帧奇慢——这是换 4K 摄像头后"卡+打不开"的元凶之一。
  - **4K UVC 常需 `MJPG`**：不设压缩格式时有些 4K/USB 机只给原始大帧或不出流；`MSMF` 后端对现代 4K UVC 通常比 `DSHOW` 更稳。
  - **无摄像头快速回退**：只有 `0 号在所有后端都连打开都失败` 才放弃(不再往后探不存在的设备号，DSHOW 探测不存在号每个卡 7-9s)。实测无摄像头约 0.08s 返回 None。
  - **硬件侧排查**(程序无能为力的情况)：直播摄像机要 USB-C 数据线接电脑(很多 C 口线只充电不传数据)、ON/OFF 拨 ON、4K@8.6W 常需插 DC 12V。先用 [scripts/probe_camera.py](scripts/probe_camera.py) 枚举 0~3 号确认 Windows 是否认到——全"打不开"就是接线/供电/驱动问题，跟程序无关。
- **`smart_tools.json` 的 `features` 字段是历史遗留**（旧 ORB descriptors，4.8MB）。YOLO 模式只读 `name` 字段（[load_tool_names](src/loader.py)）。保留 features 是为旧脚本兼容。
- **只有显示线程碰 Tkinter**：`_camera_loop` / `_static_once`（显示线程）直接调 `.config()` / `pack()` / `_show_frame()`，Tkinter 严格上非线程安全但实际稳定能用。**推理线程 `_inference_worker` 绝不能碰任何 Tkinter 控件**，只写普通 dict/list（`_last_boxes` 等）——这是双线程能安全跑的前提。要再加重负载（如额外后处理）也放推理线程，别塞进显示线程，否则视频会卡。

## 常见操作

```bash
# 验证算法（批量测试）
python tests/test_image_detect.py                       # 跑 data/samples 全部
python tests/test_image_detect.py data/samples/all_tools.jpg

# 重建 smart_tools.json（加新模板后）
python scripts/extract_features.py

# === 出包 ===
# 【推荐】一键出包：打包 + 封成 Setup.exe 一步到位
python scripts/build_all.py             # 正式版；加 --console 出调试版
# 产物：installer_output/智能工具检测系统_安装程序_v1.0.0.exe（约 200M）——发这一个文件，双击安装

# 也可分两步单独跑：
# 1) PyInstaller 打成 onedir 文件夹（自动嵌图标 assets/app_icon.ico、复制 data/yolo/best.pt）
python scripts/build_exe.py             # 正式版：--windowed 无黑窗
python scripts/build_exe.py --console   # 调试版：保留黑窗，能看 diag() 实时诊断 + run_diag.txt
# 产物：dist/智能工具检测系统/  —— 绿色免安装版，整个文件夹拷走即可双击运行（约 730M）
# 2) 用 Inno Setup 封成单文件安装程序 Setup.exe（开始菜单/桌面快捷方式带图标、可卸载）
python scripts/build_installer.py

# 注：
# - 构建会在根目录生成 智能工具检测系统.spec（PyInstaller 自动产物，已 gitignore）
# - Windows GBK 控制台跑构建脚本前先设 PYTHONIOENCODING=utf-8，否则 emoji print 会 UnicodeEncodeError
# - 换图标：覆盖 assets/app_icon.ico（或 python scripts/make_icon.py 你的图.png 重新生成）后重打
# - 瘦身：build_exe.py 里 EXCLUDES 排除了 polars(175M) 等；⚠️ matplotlib 不能排除——
#   ultralytics 在 import YOLO 时急切 import matplotlib.pyplot，排除会导致一点检测就崩
# - 装安装包工具：winget install JRSoftware.InnoSetup（ISCC.exe 在用户目录，build_installer.py 会自动找）
```
