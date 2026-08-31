"""Windows SAPI 语音播报封装"""
import time

try:
    import win32com.client as wincl
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class VoiceReporter:
    """语音播报「工具齐全」或逐个缺失工具名。

    enabled 可在播报途中被外部改为 False 实现快速中断。
    """

    def __init__(self):
        self.enabled = True
        self.speaker = None
        if _AVAILABLE:
            try:
                self.speaker = wincl.Dispatch("SAPI.SpVoice")
            except Exception as e:
                print(f"语音初始化失败: {e}")
                self.speaker = None

    def report(self, missing_tools):
        """播报缺失工具。missing_tools 为空 → 播报"工具齐全"。"""
        if self.speaker is None:
            return
        try:
            if not missing_tools:
                if self.enabled:
                    self.speaker.Speak("工具齐全")
                return
            for tool in missing_tools:
                if not self.enabled:
                    break
                self.speaker.Speak(tool + "缺失")
                time.sleep(0.3)
        except Exception as e:
            print(f"语音播报出错: {e}")
