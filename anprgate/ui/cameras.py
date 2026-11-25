"""Известные источники: недавно подключённые камеры, автоподсказки и поиск по ним."""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import List

from ..config import AppConfig
from ..video import mask_credentials

RECENTS_FILE = "data/recents.json"
MAX_RECENTS = 12


def friendly_name(uri: str) -> str:
    """Короткая подпись источника без пароля: 10.30.14.24:5345 · Ch101."""
    try:
        parsed = urllib.parse.urlparse(uri)
    except ValueError:
        return mask_credentials(uri)
    if not parsed.scheme:
        return f"{Path(uri).name or uri} (файл)"
    host = parsed.hostname or parsed.netloc
    if parsed.port:
        host = f"{host}:{parsed.port}"
    tail = [part for part in parsed.path.split("/") if part]
    label = tail[-1] if tail else ""
    if label.isdigit():
        label = f"Ch{label}"
    return f"{host} · {label}".rstrip(" ·")


def filter_sources(sources: List[str], query: str) -> List[str]:
    """Поиск по подстроке — по адресу, его подписи и замаскированному виду."""
    query = (query or "").strip().lower()
    if not query:
        return list(sources)
    hits = []
    for uri in sources:
        haystack = f"{friendly_name(uri)} {mask_credentials(uri)}".lower()
        if query in haystack:
            hits.append(uri)
    return hits


def _write(cfg: AppConfig, recents: List[str]) -> None:
    path = cfg.resolve(RECENTS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"sources": recents}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def load_recents(cfg: AppConfig) -> List[str]:
    try:
        data = json.loads(cfg.resolve(RECENTS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    sources = data.get("sources") if isinstance(data, dict) else None
    return [str(u) for u in sources if isinstance(u, str)] if sources else []


def remember_source(cfg: AppConfig, uri: str) -> List[str]:
    """Запомнить успешное подключение (вперёд списка, без дублей, не более MAX_RECENTS)."""
    uri = (uri or "").strip()
    if not uri:
        return load_recents(cfg)
    recents = [u for u in load_recents(cfg) if u != uri]
    recents.insert(0, uri)
    recents = recents[:MAX_RECENTS]
    _write(cfg, recents)
    return recents


def forget_source(cfg: AppConfig, uri: str) -> List[str]:
    recents = [u for u in load_recents(cfg) if u != uri]
    if recents:
        _write(cfg, recents)
    else:
        cfg.resolve(RECENTS_FILE).unlink(missing_ok=True)
    return recents


def _urls_in_file(path: Path) -> List[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    return [line.strip() for line in lines if line.strip().lower().startswith("rtsp://")]


def discover_seeds(cfg: AppConfig) -> List[str]:
    """Камеры из файлов cam1url.txt / cam2url.txt рядом с проектом (первичный список)."""
    seeds: List[str] = []
    seen = set()
    roots: List[Path] = []
    current = cfg.base_dir
    for _ in range(3):
        roots.append(current)
        if current.parent == current:
            break
        current = current.parent
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in ("cam*url*.txt", "cam*.url"):
            for path in sorted(root.glob(pattern)):
                for uri in _urls_in_file(path):
                    if uri not in seen:
                        seen.add(uri)
                        seeds.append(uri)
    return seeds


def known_sources(cfg: AppConfig) -> List[str]:
    """Список для диалога: недавние подключения → автоподсказки → источник из конфига."""
    out: List[str] = []
    seen = set()
    for uri in [*load_recents(cfg), *discover_seeds(cfg)]:
        if uri not in seen:
            seen.add(uri)
            out.append(uri)
    try:
        configured = cfg.source_uri()
    except Exception:
        configured = ""
    if "://" in configured and configured not in seen:
        out.append(configured)
    return out


def is_manual_uri(text: str) -> bool:
    """Строка поиска сама является адресом RTSP или существующим файлом."""
    text = (text or "").strip()
    if not text:
        return False
    return text.lower().startswith("rtsp://") or Path(text).exists()