@echo off
REM 看门狗训练启动器（供 Windows 计划任务以 SYSTEM/session0 身份调用）。
REM 放进计划任务、脱离交互桌面会话跑——这样交互桌面的显示驱动 TDR 崩溃波及不到训练进程。
cd /d "C:\Users\acad1\Desktop\python_project\auto_detection_ tool"
set PYTHONIOENCODING=utf-8
"C:\Users\acad1\AppData\Local\Programs\Python\Python311\python.exe" scripts\train_until_done.py >> "data\training\watchdog_task.log" 2>&1
