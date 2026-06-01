"""旧 ORB 算法 + 摄像头实时检测（保留对比用，已废弃）"""
import cv2
import json
import os
import numpy as np
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent / "data")

print("加载配置...")
config_file = "smart_tools.json"
with open(config_file, 'r', encoding='utf-8') as f:
    tools = json.load(f)

print("加载了 " + str(len(tools)) + " 个工具")

# 打开摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("无法打开摄像头！")
    exit()

print("\n按q退出，按空格重新检测\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    display = frame.copy()
    
    # 检测每个工具
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1000)
    kp_frame, des_frame = orb.detectAndCompute(gray, None)
    
    y_pos = 50
    detected_count = 0
    
    for tool in tools:
        if tool.get("features") is None:
            continue
        
        name = tool["name"]
        des_tool = np.array(tool["features"], dtype=np.uint8)
        
        # 匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_tool, des_frame)
        
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 50]
        match_count = len(good_matches)
        
        # 显示结果
        color = (0, 255, 0) if match_count > 15 else (0, 0, 255)
        status = "OK" if match_count > 15 else "NO"
        cv2.putText(display, status + " " + name + ": " + str(match_count), (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if match_count > 15:
            detected_count += 1
        
        y_pos += 35
    
    # 显示总数
    cv2.putText(display, "检测到: " + str(detected_count) + "/" + str(len(tools)), (20, y_pos + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    cv2.imshow("测试检测", display)
    
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

