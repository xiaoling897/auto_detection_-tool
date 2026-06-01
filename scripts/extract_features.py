"""从 data/smart_templates 和 data/samples 提取模板特征，写入 data/smart_tools.json"""
import cv2
import json
import os
import numpy as np
from pathlib import Path

# 切到 data/ 目录运行，便于使用相对路径
os.chdir(Path(__file__).resolve().parent.parent / "data")

TOOLS = [
    "钢直尺", "螺丝刀", "绝缘钢丝钳", "数字万用表",
    "绝缘测试仪", "网线测线仪", "激光测距仪",
    "数显倾角仪", "电工胶带", "磁力线坠", "胎压测试仪"
]

# 模板目录（相对 data/）
templates_dirs = ["smart_templates", "samples"]
output_file = "smart_tools.json"

# 加载已有配置
tools_data = []
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        tools_data = json.load(f)

print(f"已加载 {len(tools_data)} 个现有工具模板")

# ORB特征提取器
orb = cv2.ORB_create(nfeatures=1000)

# 处理每个模板目录
for dir_path in templates_dirs:
    if not os.path.exists(dir_path):
        continue
    
    print(f"\n处理目录: {dir_path}")
    img_files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    for img_file in img_files:
        img_path = os.path.join(dir_path, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # 尝试从文件名中识别工具名称
        tool_name = None
        for tool in TOOLS:
            if tool in img_file:
                tool_name = tool
                break
        
        if tool_name is None:
            # 如果文件名中没有工具名称，按顺序分配
            for i, tool in enumerate(TOOLS):
                if tool not in [t["name"] for t in tools_data]:
                    tool_name = tool
                    break
        
        if tool_name is None:
            continue
        
        print(f"处理: {tool_name} - {img_file}")
        
        # 提取特征
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        
        features = des.tolist() if des is not None else None
        
        # 更新或添加到tools_data
        found = False
        for i, tool in enumerate(tools_data):
            if tool["name"] == tool_name:
                tools_data[i] = {
                    "name": tool_name,
                    "template_path": img_path,
                    "features": features
                }
                found = True
                break
        
        if not found:
            tools_data.append({
                "name": tool_name,
                "template_path": img_path,
                "features": features
            })

# 保存配置
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(tools_data, f, ensure_ascii=False, indent=2)

print(f"\n已保存 {len(tools_data)} 个工具的模板！")
for tool in tools_data:
    status = "✓" if tool.get("features") is not None else "✗"
    print(f"{status} {tool['name']}")
