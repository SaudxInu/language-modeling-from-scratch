import argparse
import contextlib
import timeit
import torch
import einx
import torch.cuda.nvtx as nvtx
import numpy as np
from einops import einsum
import math

from cs336_basics.model import scaled_dot_product_attention
from cs336_basics.nn_utils import softmax


def benchmark(q, k, v, impl, only_forward, nsys, mask=None):
    if device.type == "cuda":
        synchronize = torch.cuda.synchronize
    elif device.type == "mps":
        synchronize = torch.mps.synchronize
    else:
        synchronize = lambda: None

    for _ in range(5):
        if impl == "naive":
            y_pred = scaled_dot_product_attention(q, k, v, mask)
        elif impl == "jit":
            f = torch.compile(scaled_dot_product_attention)

            y_pred = f(q, k, v, mask=mask if args.causal else None)
        else:
            raise ValueError(f"Unknown implementation: {args.impl}")

        synchronize()

        if not only_forward:
            loss = torch.nn.functional.mse_loss(y_pred, y)

            loss.backward()

            synchronize()

    cm = nvtx.range if nsys else contextlib.nullcontext

    fp_times = []
    bp_times = []
    for _ in range(100):
        with cm("forward pass"):
            start_fp = timeit.default_timer()

            if args.impl == "naive":
                y_pred = scaled_dot_product_attention(
                    q, k, v, mask=mask if args.causal else None
                )
            elif args.impl == "jit":
                f = torch.compile(scaled_dot_product_attention)

                y_pred = f(q, k, v, mask=mask if args.causal else None)
            else:
                raise ValueError(f"Unknown implementation: {args.impl}")

            synchronize()

            end_fp = timeit.default_timer()

            fp_times.append(end_fp - start_fp)

        if not args.only_forward:
            with cm("backward pass"):
                start_bp = timeit.default_timer()

                loss = torch.nn.functional.mse_loss(y_pred, y)

                loss.backward()

                synchronize()

                end_bp = timeit.default_timer()

                bp_times.append(end_bp - start_bp)

    mean_fp_time = np.mean(fp_times)
    std_fp_time = np.std(fp_times)
    print(f"    Forward pass time: {mean_fp_time:.4f} +- {std_fp_time:.4f} seconds")

    if not only_forward:
        mean_bp_time = np.mean(bp_times)
        std_bp_time = np.std(bp_times)
        print(
            f"    Backward pass time: {mean_bp_time:.4f} +- {std_bp_time:.4f} seconds"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_forward", action="store_true")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--impl", type=str, default="naive")
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument("--profile_memory", type=str, default="memory_snapshot")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.profile_memory:
        torch.cuda.memory._record_memory_history(
            enabled=True,
            trace_alloc_max_entries=1000000,
            trace_alloc_record_context=True,
        )

    batch_size = 8
    for d_model in [16, 32, 64, 128]:
        for sequence_length in [256, 1024, 4096, 8192, 16384]:
            print(f"d_model: {d_model}, sequence_length: {sequence_length}")

            q = torch.randn(
                batch_size,
                1,
                sequence_length,
                d_model,
                device=device,
                requires_grad=True,
            )
            k = torch.randn(
                batch_size,
                1,
                sequence_length,
                d_model,
                device=device,
                requires_grad=True,
            )
            v = torch.randn(
                batch_size,
                1,
                sequence_length,
                d_model,
                device=device,
                requires_grad=True,
            )
            y = torch.randn(batch_size, 1, sequence_length, d_model, device=device)

            if args.causal:
                seq = torch.arange(sequence_length, device=device)
                qi = einx.rearrange("query -> b... 1 query 1", seq, b=[1])
                kj = einx.rearrange("key   -> b... 1 1   key", seq, b=[1])
                mask = qi >= kj

            benchmark(
                q,
                k,
                v,
                args.impl,
                args.only_forward,
                args.nsys,
                mask=mask if args.causal else None,
            )

    if args.profile_memory:
        torch.cuda.memory._dump_snapshot(f"{args.profile_memory}.pickle")

        torch.cuda.memory._record_memory_history(enabled=None)
