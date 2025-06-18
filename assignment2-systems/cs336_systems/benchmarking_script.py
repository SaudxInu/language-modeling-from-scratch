import argparse
import contextlib
import timeit
import math

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from einops import einsum

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW
import cs336_basics

# TODO: Memory profiling 2.7B for 128, 256, 512 context lengths plus with mixed precision
model_configs = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def benchmark(
    model,
    optimizer,
    x,
    y,
    only_forward=False,
    warmup_steps=5,
    steps=10,
    mixed_precison=False,
    profile_memory="memory_snapshot",
    nsys=False,
):
    if device.type == "cuda":
        synchronize = torch.cuda.synchronize
    elif device.type == "mps":
        synchronize = torch.mps.synchronize
    else:
        synchronize = lambda: None

    model.train()
    for _ in range(warmup_steps):
        y_pred = model(x)

        loss = cross_entropy(y_pred, y)

        synchronize()

        if not only_forward:
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            synchronize()

    cm = (
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if mixed_precison else torch.float32,
        )
        if mixed_precison
        else contextlib.nullcontext()
    )

    cm_nsys = nvtx.range if nsys else contextlib.nullcontext

    if profile_memory:
        torch.cuda.memory._record_memory_history(
            enabled=True,
            trace_alloc_max_entries=1000000,
            trace_alloc_record_context=True,
        )

    with cm:
        fp_times = []
        bp_times = []
        for _ in range(steps):
            with cm_nsys("forward pass"):
                start_fp = timeit.default_timer()

                y_pred = model(x)

                loss = cross_entropy(y_pred, y)

                synchronize()

                end_fp = timeit.default_timer()

            if not only_forward:
                with cm_nsys("backward pass"):
                    start_bp = timeit.default_timer()

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

                    synchronize()

                    end_bp = timeit.default_timer()

                    bp_times.append(end_bp - start_bp)

            fp_times.append(end_fp - start_fp)

        if profile_memory:
            torch.cuda.memory._dump_snapshot(f"{profile_memory}.pickle")

            torch.cuda.memory._record_memory_history(enabled=None)

        mean_fp_time = np.mean(fp_times)
        std_fp_time = np.std(fp_times)
        print(f"    Forward pass time: {mean_fp_time:.4f} +- {std_fp_time:.4f} seconds")

        if not only_forward:
            mean_bp_time = np.mean(bp_times)
            std_bp_time = np.std(bp_times)
            print(
                f"    Backward pass time: {mean_bp_time:.4f} +- {std_bp_time:.4f} seconds"
            )


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]

    with nvtx.range("computing attention scores"):
        attention_scores = einsum(
            Q, K, "... query d_k, ... key d_k -> ... query key"
        ) / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range("final matmul"):
        result = einsum(
            attention_weights, V, "... query key, ... key d_v ->  ... query d_v"
        )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--only_forward", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--profile_memory", type=str, default="memory_snapshot")
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument("--jit", action="store_true")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.nsys:
        cs336_basics.model.scaled_dot_product_attention = (
            annotated_scaled_dot_product_attention
        )

    model = BasicsTransformerLM(
        vocab_size=10000,
        rope_theta=10000.0,
        context_length=args.context_length,
        **model_configs[args.model_size],
    )
    model.to(device)

    if args.jit:
        model = torch.compile(model)

    optimizer = AdamW(model.parameters())

    x = torch.randint(0, 10000, (2, args.context_length), device=device)
    y = torch.randint(0, 10000, (2, 1), device=device)

    print(
        f"Model: {model.__class__.__name__}, Config: {model_configs[args.model_size]}"
    )

    benchmark(
        model,
        optimizer,
        x,
        y,
        only_forward=args.only_forward,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        mixed_precison=args.mixed_precision,
        profile_memory=args.profile_memory,
        nsys=args.nsys,
    )
