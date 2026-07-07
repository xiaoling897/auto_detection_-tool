; ============================================================
;  智能工具检测系统 —— Inno Setup 安装脚本
;  作用：把 PyInstaller 产物 dist\智能工具检测系统\ 打包成一个
;        Setup.exe 安装程序，装进电脑、生成带图标的开始菜单/桌面
;        快捷方式，并可在「添加或删除程序」里卸载。
;
;  编译：python scripts\build_installer.py
;    （或直接用 Inno Setup 编译器打开本文件按 F9）
;  前提：先跑 python scripts\build_exe.py 生成 dist\智能工具检测系统\
; ============================================================

#define AppName "智能工具检测系统"
#define AppVersion "1.0.0"
#define AppPublisher "智能工具检测"
#define AppExe "智能工具检测系统.exe"
#define SourceDir "dist\智能工具检测系统"
#define IconFile "assets\app_icon.ico"

[Setup]
AppId={{7F3C1E42-9A6B-4C0E-BE21-2D9F5A8C4B10}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
; 安装到 Program Files 需要管理员权限（这样才算「装进电脑的正规软件」）
PrivilegesRequired=admin
DisableProgramGroupPage=yes
; Setup.exe 自身的图标 + 卸载入口图标
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
; 产物：installer_output\智能工具检测系统_安装程序_v1.0.0.exe
OutputDir=installer_output
OutputBaseFilename={#AppName}_安装程序_v{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; 64 位应用装到 64 位 Program Files
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "default"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加快捷方式："

[Files]
; 递归复制整个 PyInstaller 产物目录（含 exe、_internal 依赖、data 数据）
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
; 开始菜单
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"; IconFilename: "{app}\{#AppExe}"
Name: "{group}\卸载 {#AppName}"; Filename: "{uninstallexe}"
; 桌面（勾选 Task 才建）
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; IconFilename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
; 装完可勾选「立即运行」
Filename: "{app}\{#AppExe}"; Description: "立即运行 {#AppName}"; Flags: nowait postinstall skipifsilent

; ---- 把向导里最显眼的英文覆盖成中文（未覆盖的自动回退英文） ----
[Messages]
SetupWindowTitle=安装 - %1
SetupAppTitle=安装
WelcomeLabel1=欢迎使用 [name] 安装向导
WelcomeLabel2=即将在你的电脑上安装 [name/ver]。%n%n建议先关闭其它程序，然后点「下一步」继续。
SelectDirLabel3=安装程序会把 [name] 安装到下面的文件夹。
SelectDirBrowseLabel=点「下一步」继续；要选其它文件夹请点「浏览」。
ButtonNext=下一步(&N) >
ButtonBack=< 上一步(&B)
ButtonInstall=安装(&I)
ButtonCancel=取消
ButtonFinish=完成(&F)
ButtonBrowse=浏览(&R)...
ReadyLabel1=安装程序已准备好开始安装 [name]。
ReadyLabel2a=点「安装」开始安装。
ClickNext=点「下一步」继续。
FinishedHeadingLabel=[name] 安装完成
FinishedLabelNoIcons=安装程序已在你的电脑上装好 [name]。
FinishedLabel=安装程序已在你的电脑上装好 [name]，可通过快捷方式启动。
ClickFinish=点「完成」退出安装程序。
ConfirmUninstall=确定要彻底卸载 %1 及其全部组件吗？
UninstallStatusLabel=正在从你的电脑卸载 %1，请稍候……
