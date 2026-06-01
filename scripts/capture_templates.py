"""通过摄像头采集工具模板。在项目根目录运行：python scripts/capture_templates.py"""
import cv2
import json
import os
import numpy as np
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent / "data")

print("正在启动...")

# 工具列表
TOOLS = [
    "钢直尺", "螺丝刀", "绝缘钢丝钳", "数字万用表",
    "绝缘测试仪", "网线测线仪", "激光测距仪",
    "数显倾角仪", "电工胶带", "磁力线坠", "胎压测试仪"
]

templates_dir = "smart_templates"
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)

# 加载已有配置
config_file = "smart_tools.json"
tools_data = []
if os.path.exists(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        tools_data = json.load(f)
    print("已加载 " + str(len(tools_data)) + " 个现有工具模板")

# 打开摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("无法打开摄像头！")
    exit()

print("\n=== 工具模板采集程序 ===")
print("选择要采集的工具：")
for i, tool in enumerate(TOOLS):
    print(str(i + 1) + ". " + tool)
print("\n输入工具编号（多个用空格分开），或输入0采集全部：")

try:
    selection = input("> ").strip()
    if selection == "0":
        selected_tools = TOOLS
    else:
        indices = [int(x) - 1 for x in selection.split() if x.isdigit()]
        selected_tools = [TOOLS[i] for i in indices if 0 <= i < len(TOOLS)]
except:
    selected_tools = TOOLS

print("\n开始采集：" + ", ".join(selected_tools))
print("按空格键拍照，按n键跳过，按f键完成")
print("=" * 50)

current_tool_idx = 0
while current_tool_idx &lt; len(selected_tools):
    ret, frame = cap.read()
    if not ret:
        break
    
    tool_name = selected_tools[current_tool_idx]
    
    # 显示提示
    display = frame.copy()
    cv2.rectangle(display, (10, 10), (500, 150), (0, 0, 0), -1)
    cv2.putText(display, "请对准: " + tool_name, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display, "进度: " + str(current_tool_idx + 1) + "/" + str(len(selected_tools)), (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display, "按空格拍照，n跳过，f完成", (20, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    cv2.imshow("工具采集", display)
    
    key = cv2.waitKey(30) &amp; 0xFF
    
    if key == ord(' '):  # 空格键拍照
        # 找到工具在列表中的索引
        original_idx = TOOLS.index(tool_name)
        filename = "tool_" + str(original_idx) + "_" + tool_name + ".jpg"
        filepath = os.path.join(templates_dir, filename)
        cv2.imwrite(filepath, frame)
        
        # 提取特征
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=1000)
        kp, des = orb.detectAndCompute(gray, None)
        
        features = des.tolist() if des is not None else None
        
        # 更新或添加到tools_data
        # 先查找是否已有此工具
        found = False
        for i, tool in enumerate(tools_data):
            if tool["name"] == tool_name:
                tools_data[i] = {
                    "name": tool_name,
                    "template_path": filepath,
                    "features": features
                }
                found = True
                break
        if not found:
            tools_data.append({
                "name": tool_name,
                "template_path": filepath,
                "features": features
            })
        
        print(tool_name + " 已更新！")
        current_tool_idx += 1
        
    elif key == ord('n'):  # 跳过
        current_tool_idx += 1
        print("跳过 " + tool_name)
        
    elif key == ord('f'):  # 完成
        break

# 保存配置
with open(config_file, 'w', encoding='utf-8') as f:
    json.dump(tools_data, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 50)
print("已保存 " + str(len(tools_data)) + " 个工具的模板！")
print("现在可以在项目根目录运行 python run.py 进行检测了！")
print("=" * 50)

cap.release()
cv2.destroyAllWindows()

