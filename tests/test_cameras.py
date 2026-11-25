"""Список известных камер: имена, поиск, запоминание подключений."""

from pathlib import Path

from anprgate.config import AppConfig
from anprgate.ui.cameras import (
    MAX_RECENTS,
    discover_seeds,
    filter_sources,
    forget_source,
    friendly_name,
    is_manual_uri,
    known_sources,
    load_recents,
    remember_source,
)
from anprgate.video import ensure_source_allowed, mask_credentials

CAM1 = "rtsp://user:secret@10.30.14.24:5345/streaming/Channels/101"
CAM2 = "rtsp://user2:hunter2@10.30.0.52:5555/h264"


def make_cfg(tmp_path, base_dir=None):
    cfg = AppConfig()
    cfg.base_dir = base_dir or tmp_path
    return cfg


def test_friendly_name_hides_password():
    assert friendly_name(CAM1) == "10.30.14.24:5345 · Ch101"
    assert "secret" not in friendly_name(CAM1)
    assert friendly_name(CAM2) == "10.30.0.52:5555 · h264"


def test_friendly_name_file():
    assert friendly_name("some/clip.mp4") == "clip.mp4 (файл)"


def test_filter_sources():
    sources = [CAM1, CAM2]
    assert filter_sources(sources, "") == sources
    assert filter_sources(sources, "14.24") == [CAM1]
    assert filter_sources(sources, "ch101") == [CAM1]     # по имени
    assert filter_sources(sources, "10.30.0.52") == [CAM2]
    assert filter_sources(sources, "hunter2") == []      # пароль не является ключом поиска
    assert filter_sources(sources, "нет такого") == []


def test_remember_dedup_and_cap(tmp_path):
    cfg = make_cfg(tmp_path)
    for n in range(15):
        remember_source(cfg, f"rtsp://x:{n}@host{n}/v")
    assert len(load_recents(cfg)) == MAX_RECENTS
    recents = remember_source(cfg, CAM1)
    assert recents[0] == CAM1                      # свежая — первая
    again = remember_source(cfg, CAM1)
    assert again.count(CAM1) == 1                  # дублей нет


def test_discover_seeds_from_files(tmp_path):
    proj = tmp_path / "project"
    proj.mkdir()
    (tmp_path / "cam1url.txt").write_text(CAM1 + "\n", encoding="utf-8")
    cfg = make_cfg(tmp_path, base_dir=proj)
    assert discover_seeds(cfg) == [CAM1]
    assert known_sources(cfg) == [CAM1]


def test_forget_source(tmp_path):
    cfg = make_cfg(tmp_path)
    remember_source(cfg, CAM1)
    assert load_recents(cfg) == [CAM1]
    forget_source(cfg, CAM1)
    assert load_recents(cfg) == []


def test_is_manual_uri(tmp_path):
    assert is_manual_uri("rtsp://host:554/x")
    file = tmp_path / "clip.mp4"
    file.write_bytes(b"x")
    assert is_manual_uri(str(file))
    assert not is_manual_uri("просто текст")
    assert not is_manual_uri("")


def test_known_sources_prefers_recents(tmp_path):
    cfg = make_cfg(tmp_path)
    remember_source(cfg, CAM2)
    assert known_sources(cfg)[0] == CAM2


def test_password_is_redacted_inside_error_text():
    text = "Не удалось открыть rtsp://user:secret@10.30.14.24:5345/streaming/Channels/101"
    masked = mask_credentials(text)
    assert "secret" not in masked
    assert "10.30.14.24" in masked


def test_only_file_and_rtsp_sources_are_allowed(tmp_path):
    assert ensure_source_allowed(CAM1) == CAM1
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    assert ensure_source_allowed(str(clip)) == str(clip)
    for blocked in ("http://10.0.0.1/stream", "https://example/a", "concat:a|b", "rtsp://host\n/x"):
        try:
            ensure_source_allowed(blocked)
        except ValueError:
            continue
        raise AssertionError(blocked)