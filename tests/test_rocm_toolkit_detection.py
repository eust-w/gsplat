# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe checks for the legacy ROCm backend toolkit probe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_backend(monkeypatch):
    torch = ModuleType("torch")
    torch.__path__ = []
    torch.version = SimpleNamespace(hip=None)

    torch_utils = ModuleType("torch.utils")
    torch_utils.__path__ = []
    cpp_extension = ModuleType("torch.utils.cpp_extension")
    cpp_extension.ROCM_HOME = None
    cpp_extension._find_cuda_home = lambda: None
    torch.utils = torch_utils
    torch_utils.cpp_extension = cpp_extension

    build = ModuleType("gsplat.cuda.build")
    build.build_and_load_gsplat = lambda: object()
    gsplat = ModuleType("gsplat")
    gsplat.__path__ = []
    gsplat_cuda = ModuleType("gsplat.cuda")
    gsplat_cuda.__path__ = []
    gsplat.cuda = gsplat_cuda
    rich = ModuleType("rich")
    rich.__path__ = []
    rich_console = ModuleType("rich.console")
    rich_console.Console = lambda: SimpleNamespace(print=lambda *_args: None)

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_extension)
    monkeypatch.setitem(sys.modules, "gsplat", gsplat)
    monkeypatch.setitem(sys.modules, "gsplat.cuda", gsplat_cuda)
    monkeypatch.setitem(sys.modules, "gsplat.cuda.build", build)
    monkeypatch.setitem(sys.modules, "rich", rich)
    monkeypatch.setitem(sys.modules, "rich.console", rich_console)

    path = REPO_ROOT / "gsplat" / "cuda" / "_backend.py"
    spec = importlib.util.spec_from_file_location("gsplat.cuda._backend_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "gsplat.cuda"
    spec.loader.exec_module(module)
    return module, torch, cpp_extension


def test_rocm_toolkit_available_from_rocm_home(monkeypatch):
    module, torch, cpp_extension = _load_backend(monkeypatch)
    torch.version.hip = "7.2"
    cpp_extension.ROCM_HOME = "/opt/rocm"
    monkeypatch.setattr(
        module.os.path,
        "isfile",
        lambda path: path == "/opt/rocm/bin/hipcc",
    )

    assert module.cuda_toolkit_available()


def test_rocm_toolkit_available_from_path(monkeypatch):
    module, torch, _cpp_extension = _load_backend(monkeypatch)
    torch.version.hip = "7.2"
    monkeypatch.setattr(module.os.path, "isfile", lambda _path: False)
    monkeypatch.setattr(
        module,
        "call",
        lambda command, **_kwargs: 0 if command == ["hipcc", "--version"] else 1,
    )

    assert module.cuda_toolkit_available()


def test_rocm_toolkit_unavailable(monkeypatch):
    module, torch, _cpp_extension = _load_backend(monkeypatch)
    torch.version.hip = "7.2"
    monkeypatch.setattr(module.os.path, "isfile", lambda _path: False)

    def missing_hipcc(_command, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(module, "call", missing_hipcc)
    assert not module.cuda_toolkit_available()
