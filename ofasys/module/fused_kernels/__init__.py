# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import os
import pathlib
import subprocess
import sys

import torch

from ofasys.utils.file_utils import OFA_CACHE_HOME, local_file_lock

# Setting this param to a list has a problem of generating different
# compilation commands (with diferent order of architectures) and
# leading to recompilation of fused kernels. Set it to empty string
# to avoid recompilation and assign arch flags explicity in
# extra_cuda_cflags below
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


@local_file_lock(os.path.join(OFA_CACHE_HOME, 'fused.lock'))
def build_kernel():
    from torch.utils import cpp_extension

    # Check if cuda 11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_major, _ = _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
    if int(bare_metal_major) >= 11:
        cc_flag.append('-gencode')
        cc_flag.append('arch=compute_80,code=sm_80')

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / 'build'
    _create_build_dir(buildpath)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
        try:
            print("building fused kernel:", name)
            return cpp_extension.load(
                name=name,
                sources=sources,
                build_directory=buildpath,
                extra_cflags=['-O3'],
                extra_cuda_cflags=['-O3', '-gencode', 'arch=compute_70,code=sm_70', '--use_fast_math']
                + extra_cuda_flags
                + cc_flag,
            )
        except RuntimeError:
            print(f"loading {name} from default")
            return cpp_extension._import_module_from_library(name, srcpath / 'default_build', True)

    # ==============
    # Fused softmax.
    # ==============
    extra_cuda_flags = [
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
    ]

    # Upper triangular softmax.
    # sources=[srcpath / 'scaled_upper_triang_masked_softmax.cpp',
    #          srcpath / 'scaled_upper_triang_masked_softmax_cuda.cu']
    # scaled_upper_triang_masked_softmax_cuda = _cpp_extention_load_helper(
    #     "scaled_upper_triang_masked_softmax_cuda",
    #     sources, extra_cuda_flags)

    # Masked softmax.
    sources = [srcpath / 'scaled_masked_softmax.cpp', srcpath / 'scaled_masked_softmax_cuda.cu']
    _cpp_extention_load_helper("scaled_masked_softmax_cuda", sources, extra_cuda_flags)

    # Softmax
    sources = [srcpath / 'scaled_softmax.cpp', srcpath / 'scaled_softmax_cuda.cu']
    _cpp_extention_load_helper("scaled_softmax_cuda", sources, extra_cuda_flags)


def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")


try:
    if '--ofasys.model.unify.use_fused=True' in sys.argv:
        build_kernel()
        assert torch.cuda.is_available()
        _is_fused_kernel_available = True
        from .fused_softmax import FusedScaleMaskSoftmax

        __all__ = [
            'FusedScaleMaskSoftmax',
        ]
    else:
        _is_fused_kernel_available = False
except:  # noqa
    print('Failed to build fused kernel')
    _is_fused_kernel_available = False
