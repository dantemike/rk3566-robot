import os
import re
import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import dashscope
import pyaudio
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult


class ASRTranscriptionResult:
    """Compatibility wrapper for callbacks that previously used .text."""

    def __init__(self, sentence: Dict[str, Any], raw_result: RecognitionResult):
        self.sentence = sentence
        self.raw_result = raw_result
        self.text = str(sentence.get("text", ""))
        self.is_sentence_end = RecognitionResult.is_sentence_end(sentence)

    def get_sentence(self) -> Dict[str, Any]:
        return self.sentence


class ASRCallback(RecognitionCallback):
    """Process events from the current DashScope Recognition API."""

    def __init__(
        self,
        on_transcription: Optional[Callable] = None,
        on_translation: Optional[Callable] = None,
        on_open: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        on_sentence_end: Optional[Callable] = None,
        on_wake_word: Optional[Callable] = None,
        asr_instance=None,
    ):
        self.on_transcription = on_transcription
        # Recognition only returns ASR results. Keep this argument for old callers.
        self.on_translation = on_translation
        self.on_open_callback = on_open
        self.on_close_callback = on_close
        self.on_sentence_end = on_sentence_end
        self.on_wake_word = on_wake_word
        self.asr_instance = asr_instance
        self.mic = None
        self.stream = None

    def on_open(self) -> None:
        print("ASR连接已打开")
        self._close_microphone()
        self.mic = pyaudio.PyAudio()
        sample_rate = self.asr_instance.sample_rate if self.asr_instance else 16000
        self.stream = self.mic.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=3200,
        )
        if self.on_open_callback:
            self.on_open_callback()

    def on_close(self) -> None:
        print("ASR连接已关闭")
        self._close_microphone()
        if self.on_close_callback:
            self.on_close_callback()

    def on_error(self, result: RecognitionResult) -> None:
        request_id = self._get_request_id(result)
        message = getattr(result, "message", None) or getattr(result, "output", None)
        message = message or str(result)
        print(f"ASR识别错误: request_id={request_id}, message={message}")
        if self.asr_instance:
            self.asr_instance.last_error = str(message)
            self.asr_instance.reconnect_requested.set()

    def on_complete(self) -> None:
        print("ASR识别任务已完成")

    def on_event(self, result: RecognitionResult) -> None:
        try:
            sentence = result.get_sentence()
        except Exception as exc:
            print(f"解析ASR结果失败: {exc}")
            return

        if not isinstance(sentence, dict):
            return

        text = str(sentence.get("text", "")).strip()
        if not text:
            return

        request_id = self._get_request_id(result)
        try:
            usage = result.get_usage(sentence)
        except Exception:
            usage = None

        try:
            transcription_result = ASRTranscriptionResult(sentence, result)
        except Exception as exc:
            print(f"判断句子状态失败: {exc}")
            return

        asr = self.asr_instance
        if asr and asr.audio_paused:
            return

        clean_text = re.sub(r"[^\w\s]", "", text)
        print(f"text: {text} -> clean: {clean_text}")

        if asr:
            asr.last_audio_time = time.time()

            if not asr.wake_word_detected:
                for wake_word in asr.wake_words:
                    if wake_word.lower() in clean_text.lower():
                        print(f"检测到唤醒词: {wake_word}")
                        asr.wake_word_detected = True
                        asr.last_audio_time = time.time()
                        if self.on_wake_word:
                            self.on_wake_word(wake_word)
                        # Continue so a final wake-word sentence can reach later checks / LLM.
                        break

            if not asr.wake_word_detected:
                return

            if transcription_result.is_sentence_end and self.on_sentence_end:
                print(f"检测到句子结束: {text}")
                if asr.is_wake_only_utterance(text):
                    print(f"纯唤醒句，不送入LLM: {text}")
                    return

                if asr.should_filter_audio(text):
                    print(f"音乐播放中，过滤非音乐控制指令: {text}")
                    return
                self.on_sentence_end(text, request_id)

        if self.on_transcription:
            self.on_transcription(transcription_result, request_id, usage)

    def _close_microphone(self) -> None:
        stream = self.stream
        mic = self.mic
        self.stream = None
        self.mic = None

        if stream:
            try:
                if stream.is_active():
                    stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        if mic:
            try:
                mic.terminate()
            except Exception:
                pass

    @staticmethod
    def _get_request_id(result: RecognitionResult) -> Optional[str]:
        try:
            return result.get_request_id()
        except Exception:
            return None


class ASR:
    """Real-time speech recognition based on DashScope Recognition."""

    def __init__(
        self,
        api_key: str,
        model: str = "paraformer-realtime-v2",
        sample_rate: int = 16000,
        transcription_enabled: bool = True,
        translation_enabled: bool = False,
        translation_target_languages: Optional[List[str]] = None,
        audio_format: str = "pcm",
        workspace_id: Optional[str] = None,
        base_websocket_api_url: Optional[str] = None,
        region: str = "cn-beijing",
        heartbeat: bool = True,
        semantic_punctuation_enabled: Optional[bool] = False,
        disfluency_removal_enabled: Optional[bool] = None,
    ):
        dashscope.api_key = api_key

        self.model = model
        self.sample_rate = sample_rate
        self.transcription_enabled = transcription_enabled
        self.translation_enabled = translation_enabled
        self.translation_target_languages = translation_target_languages or ["en"]
        self.audio_format = audio_format
        self.workspace_id = workspace_id
        self.region = region
        self.heartbeat = heartbeat
        self.semantic_punctuation_enabled = semantic_punctuation_enabled
        self.disfluency_removal_enabled = disfluency_removal_enabled

        if base_websocket_api_url:
            self.base_websocket_api_url = base_websocket_api_url
        elif workspace_id:
            self.base_websocket_api_url = (
                f"wss://{workspace_id}.{region}.maas.aliyuncs.com/api-ws/v1/inference"
            )
        else:
            self.base_websocket_api_url = None

        if self.base_websocket_api_url:
            dashscope.base_websocket_api_url = self.base_websocket_api_url

        self.callback: Optional[ASRCallback] = None
        self.recognizer: Optional[Recognition] = None
        # Keep the old public attribute so other project modules do not break.
        self.translator: Optional[Recognition] = None
        self.is_running = False
        self.audio_thread: Optional[threading.Thread] = None
        self.audio_paused = False
        self.reconnect_requested = threading.Event()
        self.last_error: Optional[str] = None

        self.wake_word_detected = False
        # English name + common Chinese ASR mishearings of "lily"
        self.wake_words = [
            "lily", "lily lily", "Lily", "LILY",
            "莉莉", "丽莉", "丽丽", "哩哩", "黎黎", "雷丽", "蕾丽", "蕾莉",
            "莉ly", "蕾ly", "丽ly", "利利", "里里", "李李", "梨梨", "励励",
            # 本机麦克风常见错听
            "弟弟", "弟利", "弟丽", "迷底", "米粒", "迷你", "妮妮", "丽弟", "李弟",
            "妮丽", "妮莉", "丽妮", "哩妮", "力力", "粒粒", "哩哩哩",

            "小莫小莫", "小沫小沫", "小墨小墨", "小磨小磨", "小末小末", "小陌小陌", "小默小默",
            "小莫", "小沫", "小墨", "小磨",
            "小波小波", "小伙",
        ]
        self.last_audio_time = 0.0
        self.silence_timeout = 120

        self.music_playing = False
        self.music_filter_enabled = False
        self.music_control_keywords = [
            "关闭", "停止", "停止音乐", "关闭音乐", "关掉音乐", "停止播放", "关闭播放",
            "下一首", "下一首歌", "切歌", "换歌", "跳过", "下一曲",
            "上一首", "上一首歌", "上一曲", "前一首", "返回上一首",
            "播放音乐", "开始播放", "继续播放", "恢复播放",
            "音量", "调音量", "声音", "大小声", "静音", "取消静音",
        ]

        self.keepalive_enabled = True
        self.keepalive_interval = 0.1
        self.silence_data = b"\x00" * 3200

        if translation_enabled:
            print("提示: 新Recognition接口只负责语音识别，translation_enabled参数已保留但不再执行翻译。")

    def set_callbacks(
        self,
        on_transcription: Optional[Callable] = None,
        on_translation: Optional[Callable] = None,
        on_open: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        on_sentence_end: Optional[Callable] = None,
        on_wake_word: Optional[Callable] = None,
    ):
        self.callback = ASRCallback(
            on_transcription=on_transcription,
            on_translation=on_translation,
            on_open=on_open,
            on_close=on_close,
            on_sentence_end=on_sentence_end,
            on_wake_word=on_wake_word,
            asr_instance=self,
        )

    def start(self):
        if self.is_running:
            print("ASR服务已在运行中")
            return

        if not self.callback:
            self.callback = ASRCallback(asr_instance=self)

        self.reconnect_requested.clear()
        self.last_error = None
        recognizer = self._create_recognizer()

        try:
            recognizer.start()
            if not self._wait_for_audio_stream():
                raise RuntimeError("ASR连接已建立，但麦克风音频流未就绪")
        except Exception:
            self._close_recognizer(recognizer)
            raise

        self.recognizer = recognizer
        self.translator = recognizer
        self.is_running = True
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.audio_thread.start()
        print(f"ASR服务已启动，模型: {self.model}")

    def stop(self):
        if not self.is_running and not self.recognizer:
            print("ASR服务未在运行")
            return

        self.is_running = False
        self.reconnect_requested.clear()
        recognizer = self.recognizer
        self.recognizer = None
        self.translator = None
        self._close_recognizer(recognizer)

        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2)
        self.audio_thread = None
        print("ASR服务已停止")

    def _create_recognizer(self) -> Recognition:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "callback": self.callback,
            "format": self.audio_format,
            "sample_rate": self.sample_rate,
            "heartbeat": self.heartbeat,
        }
        if self.workspace_id:
            kwargs["workspace"] = self.workspace_id
        if self.semantic_punctuation_enabled is not None:
            kwargs["semantic_punctuation_enabled"] = self.semantic_punctuation_enabled
        if self.disfluency_removal_enabled is not None:
            kwargs["disfluency_removal_enabled"] = self.disfluency_removal_enabled
        return Recognition(**kwargs)

    def _wait_for_audio_stream(self, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.callback and self.callback.stream:
                return True
            time.sleep(0.05)
        return False

    @staticmethod
    def _close_recognizer(recognizer: Optional[Recognition]) -> None:
        if not recognizer:
            return
        try:
            recognizer.stop()
        except Exception as exc:
            print(f"关闭ASR连接时出现异常: {exc}")

    def _audio_loop(self):
        last_keepalive_time = 0.0

        while self.is_running:
            try:
                if self.reconnect_requested.is_set():
                    if not self._attempt_reconnect():
                        print("ASR连接重连失败，音频处理循环退出")
                        break
                    last_keepalive_time = time.time()
                    continue

                stream = self.callback.stream if self.callback else None
                if not stream:
                    time.sleep(0.05)
                    continue

                current_time = time.time()
                if self.wake_word_detected and self.last_audio_time > 0:
                    if current_time - self.last_audio_time > self.silence_timeout:
                        print(f"静音超时({self.silence_timeout}秒)，重置唤醒状态")
                        self.wake_word_detected = False
                        self.last_audio_time = 0
                        print("等待唤醒词...")

                if self.audio_paused:
                    try:
                        stream.read(1024, exception_on_overflow=False)
                    except Exception:
                        pass

                    if (
                        self.keepalive_enabled
                        and self.recognizer
                        and current_time - last_keepalive_time >= self.keepalive_interval
                    ):
                        self.recognizer.send_audio_frame(self.silence_data)
                        last_keepalive_time = current_time
                    time.sleep(0.05)
                    continue

                data = stream.read(3200, exception_on_overflow=False)
                if self.recognizer and self.is_running:
                    self.recognizer.send_audio_frame(data)
                    last_keepalive_time = current_time

            except Exception as exc:
                if not self.is_running:
                    break
                print(f"音频处理错误: {exc}")
                self.last_error = str(exc)
                self.reconnect_requested.set()

            time.sleep(0.01)

    def _check_network_connectivity(self) -> bool:
        host = "dashscope.aliyuncs.com"
        if self.base_websocket_api_url:
            host = urlparse(self.base_websocket_api_url).hostname or host
        try:
            with socket.create_connection((host, 443), timeout=3):
                return True
        except OSError:
            return False

    def _attempt_reconnect(self, max_retries: int = 3, retry_delay: float = 2.0) -> bool:
        self.reconnect_requested.clear()
        for attempt in range(max_retries):
            try:
                print(f"ASR重连尝试 {attempt + 1}/{max_retries}")
                if not self._check_network_connectivity():
                    print("DashScope网络连接不可用，等待恢复...")
                    time.sleep(retry_delay)
                    continue

                old_recognizer = self.recognizer
                self.recognizer = None
                self.translator = None
                self._close_recognizer(old_recognizer)
                time.sleep(retry_delay)

                recognizer = self._create_recognizer()
                recognizer.start()
                if not self._wait_for_audio_stream():
                    raise RuntimeError("重连后麦克风音频流未就绪")

                self.recognizer = recognizer
                self.translator = recognizer
                self.last_error = None
                print("ASR连接重连成功")
                return True
            except Exception as exc:
                self.last_error = str(exc)
                print(f"ASR重连失败 (尝试 {attempt + 1}/{max_retries}): {exc}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        return False

    def is_active(self) -> bool:
        return self.is_running

    def pause_audio(self):
        self.audio_paused = True
        print("ASR音频处理已暂停")

    def resume_audio(self):
        self.audio_paused = False
        print("ASR音频处理已恢复")

    def reset_wake_word(self):
        self.wake_word_detected = False
        self.last_audio_time = 0
        print("等待唤醒词...")

    def set_wake_words(self, wake_words: List[str]):
        self.wake_words = wake_words
        print("唤醒词已更新: " + ", ".join(wake_words))

    def is_wake_only_utterance(self, text: str) -> bool:
        """True if the sentence is only wake-word / name noise, not a real command."""
        clean = re.sub(r"[^\w\s]", "", text)
        prev = None
        while prev != clean:
            prev = clean
            for wake_word in sorted(self.wake_words, key=len, reverse=True):
                clean = re.sub(re.escape(wake_word), "", clean, flags=re.IGNORECASE)
            clean = re.sub(r"\s+", "", clean)
        fillers = set("的了呢啊呀哦嗯呃你我嗨哈嘿喂")
        leftover = "".join(ch for ch in clean if ch not in fillers)
        # Repeated name fragments like 丽莉丽 that survive partial stripping
        name_noise = set("丽莉哩黎lilyLily小莫沫墨磨末陌默波伙")
        leftover = "".join(ch for ch in leftover if ch not in name_noise)
        return len(leftover) == 0

    def set_keepalive_config(self, enabled: bool = True, interval: float = 0.1):
        self.keepalive_enabled = enabled
        self.keepalive_interval = interval
        print(f"保活机制配置: 启用={enabled}, 间隔={interval}秒")

    def set_music_playing(self, is_playing: bool):
        self.music_playing = is_playing
        self.music_filter_enabled = is_playing
        if is_playing:
            print("音乐开始播放，启用ASR过滤模式")
        else:
            print("音乐停止播放，关闭ASR过滤模式")

    def is_music_playing(self) -> bool:
        return self.music_playing

    def is_music_control_command(self, text: str) -> bool:
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.music_control_keywords)

    def should_filter_audio(self, text: str) -> bool:
        if not self.music_filter_enabled:
            return False
        if self.is_music_control_command(text):
            return False
        return True

    def get_connection_status(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "audio_paused": self.audio_paused,
            "wake_word_detected": self.wake_word_detected,
            "keepalive_enabled": self.keepalive_enabled,
            "keepalive_interval": self.keepalive_interval,
            "last_audio_time": self.last_audio_time,
            "silence_timeout": self.silence_timeout,
            "music_playing": self.music_playing,
            "music_filter_enabled": self.music_filter_enabled,
            "model": self.model,
            "last_error": self.last_error,
        }


if __name__ == "__main__":
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY")

    def print_transcription(transcription_result, request_id, usage):
        print(f"转录结果: {transcription_result.text}")

    def print_sentence(final_text, request_id):
        print(f"句子识别完成: {final_text}")

    asr = ASR(api_key=api_key)
    asr.set_callbacks(
        on_transcription=print_transcription,
        on_sentence_end=print_sentence,
    )

    try:
        asr.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止ASR服务...")
        asr.stop()
