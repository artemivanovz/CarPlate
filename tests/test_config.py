import pytest

from anprgate import config


def test_clone_without_local_configuration_uses_example(tmp_path, monkeypatch):
    example = tmp_path / "anpr.example.yaml"
    example.write_text("source: {uri: data/videos/example.mp4}\n", encoding="utf-8")
    monkeypatch.setattr(config, "DEFAULT_CONFIG", tmp_path / "anpr.yaml")
    monkeypatch.setattr(config, "EXAMPLE_CONFIG", example)
    monkeypatch.delenv("ANPR_SOURCE", raising=False)
    cfg = config.load_config()
    assert cfg.base_dir == tmp_path
    assert cfg.source_uri() == str(tmp_path / "data/videos/example.mp4")
    assert not (tmp_path / "anpr.yaml").exists()


def test_explicit_missing_configuration_does_not_silently_fall_back(tmp_path):
    with pytest.raises(FileNotFoundError):
        config.load_config(tmp_path / "missing.yaml")
