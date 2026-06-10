"""Windows SAPI 语音播报封装（非阻塞 + 可打断，绝不卡显示线程）。

设计：一个专属语音线程，用 SAPI 的**异步播放**（SVSFlagsAsync）把要念的内容丢给系统去念，
本线程不阻塞。任何新命令（新播报 / 停止）进来时，先用 PurgeBeforeSpeak **打断当前正在念的**
并清空 SAPI 队列，再处理新命令。所以：
  - 点「停止检测」→ stop() → 立刻静音（不管上一句念没念完）；
  - 重新检测后的播报 → report() 会先打断旧的，只念本次最新的缺失结果。
COM(SAPI) 对象在语音线程内创建/使用，避免跨线程调用 COM 出错。
"""
import queue
import threading

try:
    import pythoncom
    import win32com.client as wincl
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_ASYNC = 1        # SVSFlagsAsync：异步播放，Speak 立即返回
_PURGE = 2        # SVSFPurgeBeforeSpeak：打断当前并清空待播队列


class VoiceReporter:
    """后台线程语音播报。report()/stop() 都非阻塞；新命令会打断正在念的旧内容。"""

    def __init__(self):
        self._available = _AVAILABLE
        self._cmds = queue.Queue()
        if self._available:
            threading.Thread(target=self._worker, daemon=True).start()

    def report(self, missing_tools):
        """请求播报缺失工具（空 → 念"工具齐全"）。非阻塞，会顶掉上一条没念完的。"""
        if self._available:
            self._cmds.put(("speak", list(missing_tools)))

    def stop(self):
        """立刻停止当前播报并清空待播队列（点「停止检测」时调）。非阻塞。"""
        if self._available:
            self._cmds.put(("stop", None))

    def _worker(self):
        try:
            pythoncom.CoInitialize()
            speaker = wincl.Dispatch("SAPI.SpVoice")
        except Exception as e:
            print(f"语音初始化失败: {e}")
            return
        while True:
            action, data = self._cmds.get()          # 阻塞等命令（不占 CPU）
            # 合并积压：只保留最新一条意图，避免补念一堆过期内容
            while not self._cmds.empty():
                action, data = self._cmds.get_nowait()
            try:
                # 任何命令先打断当前正在念的 + 清空 SAPI 队列
                speaker.Speak("", _ASYNC | _PURGE)
                if action == "speak":
                    texts = ["工具齐全"] if not data else [t + "缺失" for t in data]
                    for t in texts:
                        speaker.Speak(t, _ASYNC)     # 异步排队播放，本线程不阻塞
            except Exception as e:
                print(f"语音播报出错: {e}")
