import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

AUTOTUNE_CONTIG = [triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # # Good config for fp8 inputs.
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
        #               num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
        #               num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4)
    ]

@triton.autotune(
    configs=AUTOTUNE_CONTIG,
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_l2_optimized(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               M, N, K,
               stride_xm, stride_xk,
               stride_yk, stride_yn,
               stride_om, stride_on,
               BLOCK_SIZE_M: tl.constexpr,  # Number of elements each program should process.
               BLOCK_SIZE_N: tl.constexpr,  # Number of elements each program should process.
               BLOCK_SIZE_K: tl.constexpr,  # Number of elements each program should process.
               GROUP_SIZE_M: tl.constexpr,  # Number of elements each program should process.
               ):
     # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_xm > 0)
    tl.assume(stride_xk > 0)
    tl.assume(stride_yn > 0)
    tl.assume(stride_yk > 0)
    tl.assume(stride_om > 0)
    tl.assume(stride_on > 0)

    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M # in case it doesnt divide it perfectly
    offset_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N # in case it doesnt divide it perfectly
    offset_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + ((offset_xm * stride_xm)[:, None] + offset_k[None, :] * stride_xk)
    y_ptrs = y_ptr + ((offset_k * stride_yk)[:, None] + offset_yn[None, :] * stride_yn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=offset_k[None, :] + k * BLOCK_SIZE_K < K, other=0.0)
        y = tl.load(y_ptrs, mask=offset_k[:, None] + k * BLOCK_SIZE_K < K, other=0.0)
        accumulator = tl.dot(x, y, accumulator)

        x_ptrs += BLOCK_SIZE_K * stride_xk
        y_ptrs += BLOCK_SIZE_K * stride_yk
    
    offset_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    o_ptrs = output_ptr + ((offset_om * stride_om)[:, None] + offset_on[None, :] * stride_on)
    o_mask = (offset_om[:, None] < M) & (offset_on[None, :] < N)
    tl.store(o_ptrs, accumulator.to(tl.float16), mask=o_mask)

@triton.autotune(
    configs=AUTOTUNE_CONTIG,
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               M, N, K,
               stride_xm, stride_xk,
               stride_yk, stride_yn,
               stride_om, stride_on,
               BLOCK_SIZE_M: tl.constexpr,  # Number of elements each program should process.
               BLOCK_SIZE_N: tl.constexpr,  # Number of elements each program should process.
               BLOCK_SIZE_K: tl.constexpr,  # Number of elements each program should process.
               GROUP_SIZE_M: tl.constexpr,  # Number of elements each program should process.
               ):
    pid = tl.program_id(axis=0)
    grid = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // grid
    pid_n = pid % grid

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_xm > 0)
    tl.assume(stride_xk > 0)
    tl.assume(stride_yn > 0)
    tl.assume(stride_yk > 0)
    tl.assume(stride_om > 0)
    tl.assume(stride_on > 0)

    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M # in case it doesnt divide it perfectly
    offset_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N # in case it doesnt divide it perfectly
    offset_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + ((offset_xm * stride_xm)[:, None] + offset_k[None, :] * stride_xk)
    y_ptrs = y_ptr + ((offset_k * stride_yk)[:, None] + offset_yn[None, :] * stride_yn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=offset_k[None, :] + k * BLOCK_SIZE_K < K, other=0.0)
        y = tl.load(y_ptrs, mask=offset_k[:, None] + k * BLOCK_SIZE_K < K, other=0.0)
        accumulator = tl.dot(x, y, accumulator)

        x_ptrs += BLOCK_SIZE_K * stride_xk
        y_ptrs += BLOCK_SIZE_K * stride_yk
    
    offset_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    o_ptrs = output_ptr + ((offset_om * stride_om)[:, None] + offset_on[None, :] * stride_on)
    o_mask = (offset_om[:, None] < M) & (offset_on[None, :] < N)
    tl.store(o_ptrs, accumulator.to(tl.float16), mask=o_mask)

def matmul(x: torch.Tensor, y: torch.Tensor, l2_optimized: bool = False):
    # We need to preallocate the output.
    f = matmul_kernel_l2_optimized if l2_optimized else matmul_kernel
    output = torch.empty((x.shape[0], y.shape[1]), device=DEVICE, dtype=torch.float16)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(x.shape[0], meta['BLOCK_SIZE_M']), triton.cdiv(y.shape[1], meta['BLOCK_SIZE_N']))
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    f[grid](x, y, output,
                        x.shape[0], y.shape[1], x.shape[1],
                        x.stride(0), x.stride(1),
                        y.stride(0), y.stride(1),
                        output.stride(0), output.stride(1)
                        )
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


@triton.testing.perf_report(triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            line_vals=["cublas", "triton_l2_optimized", "triton_kernel"],  # Label name for the lines
            line_names=["cublas", "Triton L2 Optimized", "Triton Kernel"],  # Line styles
            styles=[("green", "-"), ("blue", "-"), ("red", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
            args={}))
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton_l2_optimized':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, l2_optimized=True), quantiles=quantiles)
    if provider == 'triton_kernel':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, l2_optimized=False), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


torch.manual_seed(0)
M, N, K = 128, 128, 128
a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
output_torch = a @ b
output_triton = matmul(a, b)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is {torch.max(torch.abs(output_torch - output_triton))}')

benchmark.run(print_data=True, save_path="matmul-performance")

