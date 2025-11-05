import argparse
import sys
import os
import torch
import triton
import aiter
from aiter.ops.shuffle import shuffle_weight
from op_tests.triton_tests.test_gemm_afp4wfp4 import generate_gemm_afp4wfp4_inputs
from aiter.ops.triton.gemm_a16w16 import gemm_a16w16

TRITON_HIP_PRESHUFFLE_SCALES = (
    os.environ.get("TRITON_HIP_PRESHUFFLE_SCALES", "0") == "1"
)

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.general_utils import (
    torch_dtype_to_wave,
)
from wave_lang.kernel.wave.constraints import (
    MMAType,
)
from wave_lang.kernel.wave.templates.reordered_gemm import get_reordered_matmul
from aiter import hipb_mm, hipb_create_extension

def get_wave_gemm(shape, c_dtype, use_async=False):
    # Workgroup tile sizes
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    # Group size
    GROUP_SIZE_M = 16
    mfma_variant = MMAType.F32_16x16x16_F16
    reordered_gemm, hyperparams = get_reordered_matmul(
        shape[0], shape[1], shape[2], BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, mfma_variant
    )
    schedule = SchedulingType.PREFETCH
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=schedule,
        wave_runtime=True,
        dump_intermediates="./inter",
        use_buffer_ops=True,
        use_global_to_shared=use_async,
        minimize_shared_allocs=False,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, reordered_gemm)
    return gemm

def get_hipb_gemm(inp, weights, dtype):
    return hipb_mm(
        inp,
        weights.t(),
        solution_index=0,
        bias=None,
        out_dtype=dtype,
        scaleA=None,
        scaleB=None,
        scaleOut=None,
        bpreshuffle=False,
    )

def get_asm_gemm(
    x, weight, asm_out, bias=None, splitK=None, kernelName=None
):
    return aiter.gemm_a16w16_asm(x, weight, asm_out, bias, splitK, kernelName)


def run_benchmark(args):
    assert args.shape, "User can specify --shape or --model MODEL -M VAL exclusively"

    x_names = ["M", "N", "K"]
    if args.shape:
        x_vals_list = [args.shape]
    else:
        x_vals_list = get_x_vals()

    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "throughput":
        ylabel = "Throughput (TFLOPS)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth (GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    line_names = ["TFlops"]
    line_vals = ["triton"]
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name="GEMM MXFP4 x MXFP4 Benchmark",
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_gemm_afp4wfp4_blockscale(M, N, K, metric, provider):
        c_dtype = torch.float16
        x = torch.randn((M, K), device="cuda", dtype=c_dtype)
        w = torch.randn((N, K), device="cuda", dtype=c_dtype)
        # flops
        flops = 2.0 * M * N * K
        # memory transfer
        mem_read = x.numel() * x.element_size() + w.numel() * w.element_size()
        mem_write = (M * N) * 2  # TODO: Fix for c_dtype != bf16
        mem = mem_read + mem_write
        out = torch.empty(x.shape[0], w.shape[1], device=x.device, dtype=c_dtype)

        if args.backend == "wave":
            wave_shape = (M, N, K)
            gemm = get_wave_gemm(wave_shape, c_dtype, use_async=False)
            wave_out = torch.empty(M, N, device=x.device, dtype=torch.float32)
            ms = triton.testing.do_bench(
                lambda: gemm(x, w, wave_out),
                warmup=25,
                rep=100,
            )
        elif args.backend == "wave_async":
            wave_shape = (M, N, K)
            gemm = get_wave_gemm(wave_shape, c_dtype, use_async=True)
            wave_out = torch.empty(M, N, device=x.device, dtype=c_dtype)
            ms = triton.testing.do_bench(
                lambda: gemm(x, w, wave_out),
                warmup=25,
                rep=100,
            )
        elif args.backend == "triton":
            triton_out = torch.empty(M, N, device="cuda", dtype=c_dtype)
            ms = triton.testing.do_bench(
                lambda: gemm_a16w16(x, w, None, c_dtype, triton_out, activation=None),
                warmup=25,
                rep=100,
            )
        elif args.backend == "hipblas":
            hipb_create_extension()
            ms = triton.testing.do_bench(
                lambda: get_hipb_gemm(x, w, dtype=torch.bfloat16),
                warmup=25,
                rep=100,
            )
        elif args.backend == "asm":
            asm_out = torch.empty(M, N, device=x.device, dtype=torch.float32)
            ms = triton.testing.do_bench(
                lambda: get_asm_gemm(x, w, asm_out),
                warmup=25,
                rep=100,
            )

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return ms
        elif metric == "throughput":
            tflops = flops / ms * 1e-9
            return tflops
        elif metric == "bandwidth":
            bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_gemm_afp4wfp4_blockscale.run(save_path=".", print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MXFP4 x MXFP4 GEMM",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-configs",
        type=str,
        default="utils/model_configs.json",
        help="Model config json file.",
    )
    parser.add_argument(
        "-M",
        type=int,
        default=4096,
        help="M dim of model benchmark if only one model is under test",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        help="user-defined shape to benchmark",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "throughput", "bandwidth"],
        default="throughput",
        help="metric to plot",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["triton", "wave", "hipblas", "wave_async", "asm"],
        default="triton",
        help="backend to run gemm",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
