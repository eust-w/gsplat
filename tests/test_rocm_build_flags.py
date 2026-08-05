# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe checks for the ROCm-specific JIT compiler flag."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_build_module(monkeypatch, *, hip, rocm_arch=None):
    torch = ModuleType("torch")
    torch.__path__ = []
    torch.version = SimpleNamespace(hip=hip)
    torch.__config__ = SimpleNamespace(
        parallel_info=lambda: "ATen parallel backend: OpenMP"
    )

    torch_utils = ModuleType("torch.utils")
    torch_utils.__path__ = []
    cpp_extension = ModuleType("torch.utils.cpp_extension")
    cpp_extension._get_build_directory = lambda *_args, **_kwargs: "/tmp/gsplat"

    torch.utils = torch_utils
    torch_utils.cpp_extension = cpp_extension
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_extension)
    monkeypatch.delenv("NVCC_FLAGS", raising=False)
    if rocm_arch is None:
        monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    else:
        monkeypatch.setenv("PYTORCH_ROCM_ARCH", rocm_arch)

    path = REPO_ROOT / "gsplat" / "cuda" / "build.py"
    name = f"gsplat_test_runtime_build_{'hip' if hip else 'cuda'}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rocm_enables_clang_complex_builtins(monkeypatch):
    parameters = _load_build_module(monkeypatch, hip="7.2").get_build_parameters()

    assert "-D__CLANG_CUDA_COMPLEX_BUILTINS=1" in parameters.extra_cuda_cflags
    assert "-U__HIP_NO_HALF_OPERATORS__" in parameters.extra_cuda_cflags
    assert "-U__HIP_NO_HALF2_OPERATORS__" in parameters.extra_cuda_cflags
    assert "-D__hip_assert=assert" in parameters.extra_cuda_cflags
    assert "-DHIPCUB_DISABLE_BFLOAT16_SIMD_OPERATORS=1" in parameters.extra_cuda_cflags


def test_cuda_does_not_enable_rocm_clang_complex_builtins(monkeypatch):
    parameters = _load_build_module(monkeypatch, hip=None).get_build_parameters()

    assert "-D__CLANG_CUDA_COMPLEX_BUILTINS=1" not in parameters.extra_cuda_cflags


def test_rocm_rdna_enables_wave32(monkeypatch):
    parameters = _load_build_module(
        monkeypatch, hip="7.2", rocm_arch="gfx90a;gfx1100"
    ).get_build_parameters()

    assert "-D__AMDGCN_WAVEFRONT_SIZE=32" in parameters.extra_cuda_cflags


def test_rocm_cdna_keeps_wave64_default(monkeypatch):
    parameters = _load_build_module(
        monkeypatch, hip="7.2", rocm_arch="gfx90a"
    ).get_build_parameters()

    assert "-D__AMDGCN_WAVEFRONT_SIZE=32" not in parameters.extra_cuda_cflags


def test_rocm_labeled_partition_uses_tile_local_warp_intrinsics():
    source = (REPO_ROOT / "gsplat" / "cuda" / "include" / "Utils.cuh").read_text()

    assert "const LabelT src_label = __shfl(label, src, 32)" in source
    assert "g.mask = (__match_any_sync" not in source
    assert "__ballot(1) & tile_mask" in source
    assert "active_tile_mask = __activemask()" not in source
    assert "constexpr uint32_t tile_size = 32" in source
    assert "const uint32_t tile_size = warp.size()" not in source
    assert "physical_lane - tile_base" in source
    assert "warp.match_any(label)" not in source
    assert "warp.thread_rank()" not in source


def test_rocm_warp_any_is_tile_local():
    utils = (REPO_ROOT / "gsplat" / "cuda" / "include" / "Utils.cuh").read_text()
    sources = "\n".join(
        path.read_text()
        for path in (
            REPO_ROOT / "gsplat" / "cuda" / "csrc"
        ).glob("RasterizeToPixels*Bwd.cu")
    )

    assert "__ballot(predicate) & tile_mask" in utils
    assert "WARP_ANY(warp, valid)" in sources
    assert "warp.any(valid)" not in sources


def test_adam_uses_compiler_builtin_float_sqrt_on_hip():
    source = (REPO_ROOT / "gsplat" / "cuda" / "csrc" / "AdamCUDA.cu").read_text()

    assert "#if defined(USE_ROCM)" in source
    assert "__builtin_sqrtf(register_exp_avg_sq)" in source
    assert "float denom = sqrtf(register_exp_avg_sq);" in source
    assert "sqrt(register_exp_avg_sq)" not in source


def test_device_abs_does_not_require_a_runtime_math_symbol():
    utils = (REPO_ROOT / "gsplat" / "cuda" / "include" / "Utils.cuh").read_text()
    sources = "\n".join(
        path.read_text()
        for path in (REPO_ROOT / "gsplat" / "cuda" / "csrc").glob("*.cu")
    )

    assert "constexpr T gsplat_abs(T x)" in utils
    assert "return x < static_cast<T>(0) ? -x : x;" in utils
    assert "constexpr T gaussian_abs(T x)" in sources
    assert re.search(r"(?<!gsplat_)\babs\(positions\[", sources) is None
    assert re.search(r"(?<!gsplat_)\babs\(v_xy_local\.", sources) is None


def test_rocm_device_math_compatibility_is_scoped_to_gsplat():
    common = (REPO_ROOT / "gsplat" / "cuda" / "include" / "Common.h").read_text()

    assert "defined(USE_ROCM) && defined(__HIP_DEVICE_COMPILE__)" in common
    assert "__device__ constexpr T min(T a, T b)" in common
    assert "__device__ constexpr T max(T a, T b)" in common
    assert "return __builtin_sqrtf(x);" in common
    assert "return 1.0f / __builtin_sqrtf(x);" in common
