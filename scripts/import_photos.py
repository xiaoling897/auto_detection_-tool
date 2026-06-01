"""从 data/samples 把照片导入 data/smart_templates 并提取特征"""
import cv2
import json
import os
import numpy as np
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent / "data")

print("正在从照片中提取工具特征...")

photos_dir = Path("samples")
templates_dir = Path("smart_templates")

# 创建模板文件夹
if not templates_dir.exists():
    templates_dir.mkdir(parents=True)

tools_data = []

# 先扫描文件夹里的所有jpg文件
print("\n扫描文件夹 " + str(photos_dir) + "：")
photo_files = []
for file_path in photos_dir.iterdir():
    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        photo_files.append(file_path)
        print("  发现：" + file_path.name)

print("\n共发现 " + str(len(photo_files)) + " 个照片文件\n")

# 处理每个照片
for file_path in photo_files:
    # 从文件名提取工具名称（去掉后缀）
    tool_name = file_path.stem
    
    print("处理：" + tool_name)
    
    # 读取照片
    try:
        # 使用numpy从文件读取，避免编码问题
        img = cv2.imdecode(np.fromfile(str(file_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print("  无法读取照片，跳过")
            continue
    except Exception as e:
        print("  读取照片出错：" + str(e))
        continue
    
    # 复制照片到模板文件夹
    template_path = templates_dir / file_path.name
    cv2.imencode(file_path.suffix, img)[1].tofile(str(template_path))
    
    # 提取特征
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(gray, None)
    
    features = des.tolist() if des is not None else None
    
    tools_data.append({
        "name": tool_name,
        "template_path": str(template_path),
        "features": features
    })
    
    print("  已提取特征！")

# 保存配置
config_file = "smart_tools.json"
with open(config_file, 'w', encoding='utf-8') as f:
    json.dump(tools_data, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 50)
print("完成！共处理 " + str(len(tools_data)) + " 个工具！")
print("工具列表：")
for tool in tools_data:
    print("  - " + tool["name"])
print("\n现在可以在项目根目录运行 python run.py 进行检测了！")
print("=" * 50)

