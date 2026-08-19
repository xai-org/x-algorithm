# BSD 3-Clause License
#
# Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Derived from flash-attention (https://github.com/Dao-AILab/flash-attention);
# modified by X.AI Corp.

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from functools import lru_cache

import torch

try:
    from triton.tools.disasm import extract
except ImportError:
    extract = None

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import NumericMeta

StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))


load_cubin_module_data_og = cutlass.base_dsl.runtime.cuda.load_cubin_module_data
cute_compile_og = cute.compile


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
}


@lru_cache
def get_max_active_clusters(cluster_size):
    return cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size=cluster_size)


@lru_cache
def get_device_capacity(device: torch.device = None) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def assume_strides_aligned(t):
    divby = 128 // t.element_type.width
    strides = tuple(s if isinstance(s, int) else cute.assume(s, divby=divby) for s in t.stride[:-1])
    return (*strides, t.stride[-1])


def assume_tensor_aligned(t):
    if t is None:
        return None
    return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=assume_strides_aligned(t)))


def to_cute_tensor(t, assumed_align=16, leading_dim=-1, fully_dynamic=False, enable_tvm_ffi=True):
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        tensor = from_dlpack(
            t.view(torch.uint8).detach(),
            assumed_align=assumed_align,
            enable_tvm_ffi=enable_tvm_ffi,
        )
        tensor.element_type = (
            cutlass.Float8E4M3FN if t.dtype == torch.float8_e4m3fn else cutlass.Float8E5M2
        )
    else:
        tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=enable_tvm_ffi)
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)


def to_cute_aux_tensor(t, enable_tvm_ffi=True):
    assumed_align: int = getattr(t, "__assumed_align__", None)
    leading_dim: int = getattr(t, "__leading_dim__", None)
    fully_dynamic: bool = leading_dim is None

    return to_cute_tensor(
        t,
        assumed_align=assumed_align,
        leading_dim=leading_dim,
        fully_dynamic=fully_dynamic,
        enable_tvm_ffi=enable_tvm_ffi,
    )


def get_aux_tensor_metadata(aux_tensors):
    return tuple(
        (
            getattr(t, "__assumed_align__", 0),
            getattr(t, "__leading_dim__", -1),
            hasattr(t, "__leading_dim__"),
        )
        for t in aux_tensors
    )


def get_broadcast_dims(tensor: torch.Tensor) -> tuple[bool, ...]:
    return tuple(s == 0 for s in tensor.stride())


def dump_kernel_attributes(compiled_kernel):
    import torch
    from cuda.bindings import driver
    from cutlass.utils import HardwareInfo

    device_id = torch.cuda.current_device()
    hardware_info = HardwareInfo(device_id=device_id)
    cubin_data = compiled_kernel.artifacts.CUBIN
    assert cubin_data is not None, "cubin_data is None, need '--keep-cubin' option when compiling"
    cuda_library = hardware_info._checkCudaErrors(
        driver.cuLibraryLoadData(cubin_data, None, None, 0, None, None, 0)
    )
    kernels = hardware_info._checkCudaErrors(driver.cuLibraryEnumerateKernels(1, cuda_library))
    kernel = hardware_info._checkCudaErrors(driver.cuKernelGetFunction(kernels[0]))
    local_size_bytes = hardware_info._checkCudaErrors(
        driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            kernel,
        )
    )
    num_regs = hardware_info._checkCudaErrors(
        driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
            kernel,
        )
    )

    print("--- Kernel Info ---")
    print(f"local_size_bytes: {local_size_bytes}")
    print(f"num_regs: {num_regs}")
    print("--- End Kernel Info ---")
