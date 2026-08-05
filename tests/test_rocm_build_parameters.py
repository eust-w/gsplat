# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-safe coverage for CUDA versus ROCm JIT build parameters."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_build_module(monkeypatch, *, hip):
    torch = ModuleType("torch")
    torch.__path__ = []
    torch.version = SimpleNamespace(hip=hip)
    torch.__config__ = SimpleNamespace(
        parallel_info=lambda: "ATen parallel backend: OpenMP"
    )

    torch_utils = ModuleType("torch.utils")
    torch_utils.__path__ = []
    cpp_extension = ModuleType("torch.utils.cpp_extension")
    cpp_extension.CUDA_HOME = None
    cpp_extension._get_build_directory = lambda *_args, **_kwargs: "/tmp/gsplat"

    torch.utils = torch_utils
    torch_utils.cpp_extension = cpp_extension
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_extension)
    for name in (
        "DEBUG",
        "FAST_MATH",
        "WITH_SYMBOLS",
        "NVCC_FLAGS",
        "NUM_CHANNELS",
    ):
        monkeypatch.delenv(name, raising=False)

    path = REPO_ROOT / "gsplat" / "cuda" / "build.py"
    module_name = f"gsplat_test_build_{'hip' if hip else 'cuda'}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("hip", ["7.2", "6.3"])
def test_rocm_build_uses_clang_flags_and_cuda_header_path(monkeypatch, hip):
    module = _load_build_module(monkeypatch, hip=hip)
    parameters = module.get_build_parameters()

    assert str(REPO_ROOT / "gsplat" / "cuda" / "csrc") in {
        str(Path(path).resolve()) for path in parameters.extra_include_paths
    }
    assert "--forward-unknown-opts" not in parameters.extra_cuda_cflags
    assert "-diag-suppress" not in parameters.extra_cuda_cflags
    assert "-use_fast_math" not in parameters.extra_cuda_cflags
    assert "-ffast-math" in parameters.extra_cuda_cflags
    assert "-DUSE_ROCM" in parameters.extra_cflags


def test_cuda_build_keeps_nvcc_flags(monkeypatch):
    module = _load_build_module(monkeypatch, hip=None)
    parameters = module.get_build_parameters()

    assert "--forward-unknown-opts" in parameters.extra_cuda_cflags
    assert "-diag-suppress" in parameters.extra_cuda_cflags
    assert "-use_fast_math" in parameters.extra_cuda_cflags
    assert "-ffast-math" not in parameters.extra_cuda_cflags
