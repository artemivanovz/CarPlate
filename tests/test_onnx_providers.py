import sys
from types import SimpleNamespace

import pytest

from anprgate.models import plate_reader


def test_explicit_cpu_does_not_initialize_cuda_libraries(monkeypatch):
    monkeypatch.setattr(plate_reader, "_expose_cuda_dlls", lambda: pytest.fail("unexpected CUDA initialization"))
    assert plate_reader.onnx_providers("cpu") == ["CPUExecutionProvider"]


@pytest.mark.parametrize("gpu_available, cuda_provider", [(True, True), (False, True), (True, False)])
def test_cuda_libraries_are_preloaded_only_for_a_usable_gpu(monkeypatch, gpu_available, cuda_provider):
    calls = []
    providers = ["CPUExecutionProvider"] + (["CUDAExecutionProvider"] if cuda_provider else [])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: gpu_available)))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        get_available_providers=lambda: providers, preload_dlls=lambda: calls.append("preload")))
    monkeypatch.setattr(plate_reader, "_expose_cuda_dlls", lambda: calls.append("paths"))
    selected = plate_reader.onnx_providers("auto")
    if gpu_available and cuda_provider:
        assert selected == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert calls == ["paths", "preload"]
    else:
        assert selected == ["CPUExecutionProvider"]
        assert calls == []
