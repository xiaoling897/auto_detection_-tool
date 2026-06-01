"""调试用：实时显示每个工具的 ORB 匹配数（旧算法，仅作对比参考）"""
import cv2
import json
import os
import numpy as np
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent / "data")

config_file = "smart_tools.json"
with open(config_file, 'r', encoding='utf-8') as f:
    tools = json.load(f)

print(f"共加载 {len(tools)} 个工具配置")

# ORB特征提取
orb = cv2.ORB_create(nfeatures=1000)

# 打开摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("摄像头已打开，按 'q' 退出")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 提取当前帧特征
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

    if des_frame is None:
        cv2.imshow('Debug Detection', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        continue

    # 检测每个工具
    match_results = []
    for tool in tools:
        if tool.get("features") is None:
            continue

        try:
            des_tool = np.array(tool["features"], dtype=np.uint8)

            # 匹配特征
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des_tool, des_frame)

            if len(matches) > 0:
                matches = sorted(matches, key=lambda x: x.distance)
                good_matches = [m for m in matches if m.distance < 65]
                match_count = len(good_matches)
                match_results.append((tool["name"], match_count))
        except Exception as e:
            pass

    # 打印结果
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 60)
    print("当前匹配结果（距离阈值65）：")
    print("-" * 60)
    for name, count in match_results:
        status = "✓ 检测到" if count > 6 else "✗ 未检测到"
        print(f"{name:<15} 匹配数: {count:<3} {status}")
    print("-" * 60)
    print(f"共检测到 {sum(1 for _, c in match_results if c > 6)} / {len([t for t in tools if t.get('features') is not None])} 个工具")
    print("=" * 60)
    print("按 'q' 退出")

    # 显示画面
    cv2.imshow('Debug Detection', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
