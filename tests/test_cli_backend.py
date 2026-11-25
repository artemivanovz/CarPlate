import pytest

from anprgate import cli


@pytest.mark.parametrize("command", ["ui", "headless"])
@pytest.mark.parametrize("backend", [None, "fast_alpr", "ru_anpr"])
def test_runtime_backend_override_preserves_configuration(tmp_path, monkeypatch, command, backend):
    config = tmp_path / "anpr.yaml"
    original = "models: {plate: {backend: ru_anpr}}\n"
    config.write_text(original, encoding="utf-8")
    observed = []
    monkeypatch.setattr(cli, "cmd_" + command, lambda args, cfg: observed.append(cfg.models.plate.backend) or 0)
    arguments = ["--config", str(config), command]
    if backend:
        arguments += ["--plate-backend", backend]
    assert cli.main(arguments) == 0
    assert observed == [backend or "ru_anpr"]
    assert config.read_text(encoding="utf-8") == original
