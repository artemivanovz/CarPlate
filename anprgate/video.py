"""Чтение источников видео: mp4-файлы и RTSP-потоки камер.

RTSP читается в отдельном потоке-«пожирателе»: он непрерывно декодирует поток
в штатном темпе камеры и держит только последний кадр. Пайплайн забирает самый
свежий кадр и никогда не накапливает отставание.

Живое видео идёт через PyAV (свежий FFmpeg). Встроенный в OpenCV FFmpeg 4.4
на 4K HEVC рвёт NAL-пакеты: кадр дорисовывается до половины, остальное зелёное
или серое, а декодер жжёт CPU на починке битого потока. «Линия» этого не делает,
потому что собирает тот же RTSP своим декодером.

Управление через переменные окружения:
    ANPR_RTSP_TRANSPORT        tcp | udp          (по умолчанию tcp)
    ANPR_RTSP_OPEN_TIMEOUT_MS  число              (по умолчанию 8000)
    ANPR_RTSP_RECONNECT_SEC    число              (по умолчанию 2)
    ANPR_HW_ACCEL              auto | off         (по умолчанию auto: D3D11, иначе CPU)
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple
from urllib.parse import quote_from_bytes, unquote_to_bytes, urlsplit, urlunsplit

import cv2
import numpy as np

try:
    import av
    from av.codec.hwaccel import HWAccel
except ImportError:  # пакет av ставится из requirements.txt
    av = None
    HWAccel = None

log = logging.getLogger(__name__)

_FF_OPTS_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
_URL_RE = re.compile(
    r"\b[a-z][a-z0-9+.\-]*://(?:(?!\b[a-z][a-z0-9+.\-]*://)[^\r\n<>])+",
    re.IGNORECASE,
)

# Журнал декодера не тащим в консоль: свои обрывы пишем сами.
# Вернуть строки FFmpeg: OPENCV_FFMPEG_LOGLEVEL=16 и ANPR_FFMPEG_LOG=1.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
if av is not None and os.environ.get("ANPR_FFMPEG_LOG") != "1":
    av.logging.set_level(av.logging.PANIC)


def mask_credentials(url: str) -> str:
    """Скрывает пароль (и логин без пароля) в URL — для журналов, ошибок и БД."""
    if "@" not in url:
        return url

    def _sub(match: re.Match) -> str:
        scheme, rest = match.group().split("://", 1)
        end = re.search(r"[/?#]", rest)
        authority, tail = (rest[:end.start()], rest[end.start():]) if end else (rest, "")
        if "@" not in authority:
            if ":" in authority and "@" in tail:
                # Malformed userinfo may contain a raw /, ? or #. Fail closed.
                return f"{scheme}://***"
            return match.group()
        userinfo, host = authority.rsplit("@", 1)
        user, separator, _password = userinfo.partition(":")
        auth = f"{user or '***'}:***" if separator else "***"
        return f"{scheme}://{auth}@{host}{tail}"

    return _URL_RE.sub(_sub, url)


def ensure_source_allowed(uri: str) -> str:
    """Файл на диске или RTSP. Прочие схемы (http, concat ffmpeg) не открываем."""
    text = (uri or "").strip()
    if not text or any(char in text for char in "\r\n\x00"):
        raise ValueError("Недопустимый адрес источника")
    if "://" in text:
        scheme = text.split("://", 1)[0].lower()
        if scheme not in {"rtsp", "rtsps", "file"}:
            raise ValueError(f"Разрешены только локальный файл и RTSP, схема «{scheme}» отклонена")
        return text
    # concat:, subfile: и прочие протоколы ffmpeg — не пути Windows (C:\...).
    if re.match(r"^[a-z][a-z0-9+.\-]*:", text.lower()) and not re.match(r"^[a-z]:[\\/]", text.lower()):
        raise ValueError("Разрешены только локальный файл и RTSP")
    return text


@dataclass(frozen=True)
class Frame:
    index: int
    ts: float                 # секунды: файл — позиция медиа, поток — локальное время
    image: np.ndarray


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _is_rtsp(uri: str) -> bool:
    scheme = uri.split("://", 1)[0].lower() if "://" in uri else ""
    return scheme in {"rtsp", "rtsps"}


def _hw_accel_enabled() -> bool:
    return (os.environ.get("ANPR_HW_ACCEL") or "auto").strip().lower() not in {"off", "cpu", "no", "0"}


def quote_rtsp_userinfo(uri: str) -> str:
    """Кодирует логин и пароль. Символ & в пароле иначе обрывает адрес для FFmpeg."""
    if not _is_rtsp(uri):
        return uri
    parts = urlsplit(uri)
    if parts.username is None and parts.password is None:
        return uri
    # Preserve already escaped bytes, including non-UTF-8 passwords, exactly once.
    user = quote_from_bytes(unquote_to_bytes(parts.username or ""), safe="")
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    password = quote_from_bytes(unquote_to_bytes(parts.password), safe="") if parts.password is not None else None
    auth = user if password is None else f"{user}:{password}"
    netloc = f"{auth}@{host}"
    if parts.port is not None:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))


def rtsp_ffmpeg_options() -> dict:
    """Параметры приёма, при которых пакеты кадра не выбрасываются по дороге."""
    transport = (os.environ.get("ANPR_RTSP_TRANSPORT") or os.environ.get("RTSP_TRANSPORT") or "tcp").strip().lower()
    open_timeout_ms = max(1000.0, _env_float("ANPR_RTSP_OPEN_TIMEOUT_MS", 8000.0))
    timeout_us = str(int(open_timeout_ms * 1000))
    return {
        "rtsp_transport": transport,
        "stimeout": timeout_us,
        "rw_timeout": timeout_us,
        "buffer_size": "8388608",
        "max_delay": "500000",
        "allowed_media_types": "video",
        "probesize": "5000000",
        "analyzeduration": "2000000",
    }


def _configure_ffmpeg_env(uri: str) -> None:
    """Настраивает OPENCV_FFMPEG_CAPTURE_OPTIONS под конкретный источник.

    OpenCV читает переменную при каждом открытии capture, поэтому опции RTSP
    ставим только для rtsp://, а для файлов переменную снимаем.
    """
    pairs: dict = {}
    user_value = os.environ.get(_FF_OPTS_ENV)
    if user_value:
        for part in user_value.split("|"):
            if ";" in part:
                key, value = part.split(";", 1)
                pairs[key.strip()] = value.strip()
    if _is_rtsp(uri):
        for key, value in rtsp_ffmpeg_options().items():
            pairs.setdefault(key, value)
        options = rtsp_ffmpeg_options()
        pairs["rtsp_transport"] = options["rtsp_transport"]
        pairs["stimeout"] = options["stimeout"]
        os.environ[_FF_OPTS_ENV] = "|".join(f"{key};{value}" for key, value in pairs.items())
    else:
        os.environ.pop(_FF_OPTS_ENV, None)


def _open_capture(uri: str, hw: bool) -> Optional[Tuple[cv2.VideoCapture, str]]:
    """Открывает файл. Для RTSP аппаратный путь OpenCV не используем: он рисует полосы."""
    _configure_ffmpeg_env(uri)
    open_uri = quote_rtsp_userinfo(uri)
    attempts: list = []
    if hw and _hw_accel_enabled() and not _is_rtsp(uri):
        accel_any = int(getattr(cv2, "VIDEO_ACCELERATION_ANY", 1))
        attempts.append(("hw", [cv2.CAP_PROP_HW_ACCELERATION, accel_any]))
    if _is_rtsp(uri):
        attempts.append(("sw", [cv2.CAP_PROP_HW_ACCELERATION, int(getattr(cv2, "VIDEO_ACCELERATION_NONE", 0))]))
    else:
        attempts.append(("sw", []))
    for kind, params in attempts:
        try:
            if params:
                cap = cv2.VideoCapture(open_uri, cv2.CAP_FFMPEG, params)
            else:
                cap = cv2.VideoCapture(open_uri, cv2.CAP_FFMPEG)
        except Exception:
            log.exception("Не удалось создать VideoCapture: %s", mask_credentials(uri))
            continue
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            continue
        if kind == "hw":
            mode = int(cap.get(cv2.CAP_PROP_HW_ACCELERATION) or 0)
            if mode == 0:
                cap.release()
                continue
        return cap, kind
    return None


class VideoSource:
    """Источник кадров: потоки читает параллельный поток-пожиратель без очередей."""

    def __init__(self, uri: str, loop: bool = False, realtime: bool = True) -> None:
        self.uri = ensure_source_allowed(uri)
        self.loop = bool(loop)
        self.realtime = bool(realtime)
        self.display_name = mask_credentials(uri)
        self.is_stream = bool(
            re.match(r"^[a-z][a-z0-9+.\-]*://", uri.lower())
            and not uri.lower().startswith("file://")
        )
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._seq = 0
        self._last_image: Optional[np.ndarray] = None
        self._rebase_clock = False
        self._reader: Optional[threading.Thread] = None
        if self.is_stream:
            self._reader = threading.Thread(target=self._reader_loop, name="anpr-reader", daemon=True)
            self._reader.start()

    # --- Публичный интерфейс ---------------------------------------------

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def stop(self, *, wait: bool = True) -> None:
        self._stop_event.set()
        reader = self._reader
        if wait and reader is not None and reader.is_alive() and reader is not threading.current_thread():
            reader.join(timeout=5.0)

    def reset_clock(self) -> None:
        """После паузы темп файла отсчитывается заново от текущего кадра."""
        self._rebase_clock = True

    def frames(self) -> Iterator[Frame]:
        if self.is_stream:
            yield from self._frames_stream()
        else:
            yield from self._frames_file()

    # --- Потоки (RTSP) ----------------------------------------------------

    def _reader_loop(self) -> None:
        if av is not None:
            self._read_pyav()
            return
        log.warning(
            "Пакет av не установлен — поток читает старый декодер OpenCV. "
            "На 4K HEVC кадр будет с битыми блоками. Нужен pip install av"
        )
        self._read_opencv()

    def _read_pyav(self) -> None:
        """Приём RTSP декодером PyAV: пакеты кадра собираются целиком."""
        reconnect = max(0.5, _env_float("ANPR_RTSP_RECONNECT_SEC", 2.0))
        transport = rtsp_ffmpeg_options()["rtsp_transport"]
        delay = 0.0
        use_hw = _hw_accel_enabled() and os.name == "nt" and HWAccel is not None
        report_every = max(5.0, _env_float("ANPR_READER_REPORT_SEC", 30.0))
        next_report = time.monotonic() + report_every
        reported = 0
        reported_at = time.monotonic()
        open_timeout = max(5.0, _env_float("ANPR_RTSP_OPEN_TIMEOUT_MS", 8000.0) / 1000.0)
        while not self._stop_event.is_set():
            if delay and self._stop_event.wait(delay):
                break
            hw = None
            if use_hw:
                try:
                    hw = HWAccel("d3d11va", allow_software_fallback=False)
                except Exception:
                    use_hw = False
            container = None
            decoded = 0
            kind = "d3d11" if hw is not None else "cpu"
            try:
                container = av.open(
                    quote_rtsp_userinfo(self.uri),
                    mode="r",
                    options=rtsp_ffmpeg_options(),
                    timeout=(open_timeout, 5.0),
                    hwaccel=hw,
                )
                stream = container.streams.video[0]
                if hw is None:
                    try:
                        stream.thread_type = "AUTO"
                    except Exception:
                        pass
                log.info(
                    "RTSP %s: приём %s, %dx%d %s, транспорт %s",
                    self.display_name,
                    kind,
                    int(stream.codec_context.width or 0),
                    int(stream.codec_context.height or 0),
                    stream.codec_context.name,
                    transport,
                )
                delay = 0.0
                for frame in container.decode(stream):
                    if self._stop_event.is_set():
                        break
                    image = frame.to_ndarray(format="bgr24")
                    decoded += 1
                    with self._lock:
                        self._last_image = image
                        self._seq += 1
                    now = time.monotonic()
                    if now >= next_report:
                        rate = (decoded - reported) / max(0.1, now - reported_at)
                        log.info(
                            "RTSP %s: декодер %.1f кадров/с, прочитано %d, %s",
                            self.display_name, rate, decoded, kind,
                        )
                        reported, reported_at, next_report = decoded, now, now + report_every
            except Exception as exc:
                if not self._stop_event.is_set():
                    log.warning("RTSP %s: %s", self.display_name, mask_credentials(str(exc)))
            finally:
                if container is not None:
                    try:
                        container.close()
                    except Exception:
                        pass
            if self._stop_event.is_set():
                break
            if decoded == 0 and use_hw:
                use_hw = False
                log.warning("Аппаратный декодер %s не дал кадров — дальше CPU", self.display_name)
                delay = 0.0
                continue
            if decoded == 0:
                delay = min(1.5 * reconnect, 30.0)
                continue
            log.warning("Поток %s оборвался после %d кадров, переподключение", self.display_name, decoded)
            delay = reconnect

    def _read_opencv(self) -> None:
        reconnect = max(0.5, _env_float("ANPR_RTSP_RECONNECT_SEC", 2.0))
        transport = (os.environ.get("ANPR_RTSP_TRANSPORT") or os.environ.get("RTSP_TRANSPORT") or "tcp").strip().lower()
        delay = 0.0
        hw_attempt = True
        report_every = max(5.0, _env_float("ANPR_READER_REPORT_SEC", 30.0))
        next_report = time.monotonic() + report_every
        reported = 0
        reported_at = time.monotonic()
        while not self._stop_event.is_set():
            if delay:
                if self._stop_event.wait(delay):
                    break
            opened = _open_capture(self.uri, hw=hw_attempt)
            if opened is None:
                log.warning("RTSP открыть не удалось (%s), повтор через %.1f с", self.display_name, reconnect)
                delay = min(1.5 * reconnect, 30.0)
                continue
            cap, kind = opened
            delay = 0.0
            decoded, failures = 0, 0
            while not self._stop_event.is_set():
                ok, image = cap.read()
                if not ok:
                    failures += 1
                    if failures >= 2 and decoded == 0 and kind == "hw":
                        hw_attempt = False      # больше не считаем декодер аппаратным
                        log.warning("Аппаратный декодер %s не даёт кадров — откат на CPU", self.display_name)
                        break
                    if failures >= 3:
                        break                   # обрыв/таймаут потока → переподключение
                    continue
                failures, decoded = 0, decoded + 1
                # read() может писать в тот же буфер: отдаём копию, иначе кадр рвётся.
                published = image.copy()
                with self._lock:
                    self._last_image = published
                    self._seq += 1
                now = time.monotonic()
                if now >= next_report:
                    rate = (decoded - reported) / max(0.1, now - reported_at)
                    log.info("RTSP %s: декодер %.1f кадров/с, прочитано %d, транспорт %s",
                             self.display_name, rate, decoded, transport)
                    reported, reported_at, next_report = decoded, now, now + report_every
            cap.release()
            if self._stop_event.is_set():
                break
            if decoded == 0:
                delay = 0.2                     # мгновенный ретрай (не считая hw-отката, он уже учтён)
                continue
            log.warning("Поток %s оборвался после %d кадров, переподключение", self.display_name, decoded)
            delay = reconnect

    def _frames_stream(self) -> Iterator[Frame]:
        seen = self._seq
        while not self._stop_event.is_set():
            with self._lock:
                seq, image = self._seq, self._last_image
            if image is None or seq == seen:
                time.sleep(0.01)
                continue
            seen = seq
            yield Frame(index=seq, ts=time.time(), image=image)

    # --- Файлы (mp4) ------------------------------------------------------

    def _frames_file(self) -> Iterator[Frame]:
        index = 0
        while not self._stop_event.is_set():
            opened = _open_capture(self.uri, hw=False)
            if opened is None:
                raise RuntimeError(f"Источник не открывается: {self.display_name}")
            cap = opened[0]
            try:
                anchor: Optional[float] = None
                ok, image = cap.read()
                while ok and not self._stop_event.is_set():
                    pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                    ts = pos if pos > 0 else index / fps if fps > 0 else time.time()
                    if self.realtime:
                        anchor = self._pace(anchor, ts)
                    index += 1
                    yield Frame(index - 1, float(ts), image)
                    ok, image = cap.read()
            finally:
                cap.release()
            if not self.loop:
                break
            log.info("Источник %s закончился, запускаю заново (loop)", self.display_name)

    def _pace(self, anchor: Optional[float], ts: float) -> float:
        """Удерживает файловый источник в реальном темпе; пауза перебазирует точку отсчёта."""
        if anchor is None or self._rebase_clock:
            self._rebase_clock = False
            return time.perf_counter() - ts
        wake = anchor + ts
        delay = wake - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        return anchor
