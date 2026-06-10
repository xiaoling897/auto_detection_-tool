"""Windows SAPI 语音播报封装（非阻塞：独立后台线程念，绝不卡显示线程）。

旧版 report() 在调用线程里同步 Speak（每个工具还 sleep 0.3s），跑在 GUI 显示线程里会把
视频卡住——打开视频时一次念 9 个缺失工具能冻好几秒、之后每 8s 一卡。现在改成：
report() 只把"要念的清单"丢进队列（瞬间返回），由一个专属语音线程取出来念。
COM(SAPI) 对象在它自己的线程里创建/使用，避免跨线程调用 COM 的报错。
"""
import queue
import threading

try:
    import pythoncom
    import win32com.client as wincl
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class VoiceReporter:
    """后台线程语音播报。report() 非阻塞；上一轮还在念时新请求会被丢弃，避免堆积。"""

    def __init__(self):
        self.enabled = True
        self._available = _AVAILABLE
        self._queue = queue.Queue(maxsize=1)   # 只保留一条待播报，过期的直接丢
        if self._available:
            threading.Thread(target=self._worker, daemon=True).start()

    def report(self, missing_tools):
        """非阻塞：把播报请求塞进队列即返回。队列满(上一轮还在念)就跳过本次。"""
        if not self._available:
            return
        try:
            self._queue.put_nowait(list(missing_tools))
        except queue.Full:
            pass   # 上一轮还没念完 → 本次跳过，绝不阻塞调用方(显示线程)

    def _worker(self):
        """专属语音线程：在本线程初始化 COM + 创建 SAPI，循环取队列念。"""
        try:
            pythoncom.CoInitialize()
            speaker = wincl.Dispatch("SAPI.SpVoice")
        except Exception as e:
            print(f"语音初始化失败: {e}")
            return
        while True:
            missing = self._queue.get()        # 阻塞等下一次请求（不占 CPU）
            try:
                if not missing:
                    if self.enabled:
                        speaker.Speak("工具齐全")
                    continue
                for tool in missing:
                    if not self.enabled:
                        break
                    speaker.Speak(tool + "缺失")
            except Exception as e:
                print(f"语音播报出错: {e}")
