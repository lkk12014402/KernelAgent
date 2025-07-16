# ======== matadd with on-host Tensor Memory Accelerator (TMA) integration ==========
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(
    x_desc,
    y_desc,
    output_desc,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    x = x_desc.load([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N])
    y = y_desc.load([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N])
    output = x + y
    output_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], output)


def add(x: torch.Tensor, y: torch.Tensor):
    M, N = x.shape
    output = torch.empty((M, N), device=x.device, dtype=torch.float16)

    # TMA descriptors for loading A, B and storing C
    x_desc = TensorDescriptor(x, x.shape, x.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_N])
    y_desc = TensorDescriptor(y, y.shape, y.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_N])
    output_desc = TensorDescriptor(
        output, output.shape, output.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_N]
    )

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    add_kernel[grid](
        x_desc,
        y_desc,
        output_desc,
        M,
        N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return output
