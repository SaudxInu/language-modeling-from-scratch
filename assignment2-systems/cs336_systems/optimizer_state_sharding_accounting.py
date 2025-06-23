# torchrun --nnodes=1 --nproc_per_node=2 cs336_systems/optimizer_state_sharding_accounting.py --model_size large --profile_memory
import argparse
import timeit
import contextlib

import numpy as np
import torch
import os
import torch
import torch.distributed as dist
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.optimizer_state_sharding import ShardedOptimizer


model_configs = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def setup():
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup():
    dist.destroy_process_group()


def run(
    model,
    optimizer,
    x,
    y,
    nsys,
):
    if device.type == "cuda":
        synchronize = torch.cuda.synchronize
    elif device.type == "mps":
        synchronize = torch.mps.synchronize
    else:
        synchronize = lambda: None

    cm_nsys = nvtx.range if nsys else contextlib.nullcontext

    start = timeit.default_timer()

    with cm_nsys("forward_compute"):
        y_pred = model(x)

        loss = cross_entropy(y_pred, y)

        # synchronize()

    with cm_nsys("backward_compute"):
        optimizer.zero_grad()

        loss.backward()

        # synchronize()

    with cm_nsys("backward_comm"):
        start_comm = timeit.default_timer()

        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= int(os.environ["WORLD_SIZE"])

        # synchronize()

    memory = torch.cuda.memory_allocated(device="cuda")
    print(
        f"Memory allocated after backward pass on device {int(os.environ['LOCAL_RANK'])}: {memory / (1024 ** 2):.2f} MB"
    )

    end_comm = timeit.default_timer()

    optimizer.step()

    memory = torch.cuda.memory_allocated(device="cuda")
    print(
        f"Memory allocated after optimizer step on device {int(os.environ['LOCAL_RANK'])}: {memory / (1024 ** 2):.2f} MB"
    )

    synchronize()

    end = timeit.default_timer()

    return end - start, end_comm - start_comm


def main(context_length, model_size, steps, nsys, shard_optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup()

    memory = torch.cuda.memory_allocated(device="cuda")
    print(
        f"Initial memory allocated on device {int(os.environ['LOCAL_RANK'])}: {memory / (1024 ** 2):.2f} MB"
    )

    model = BasicsTransformerLM(
        vocab_size=10000,
        rope_theta=10000.0,
        context_length=context_length,
        **model_configs[model_size],
    )
    model = model.to(device)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    memory = torch.cuda.memory_allocated(device="cuda")
    print(
        f"Memory allocated after model initialization on device {int(os.environ['LOCAL_RANK'])}: {memory / (1024 ** 2):.2f} MB"
    )

    if not shard_optimizer:
        optimizer = AdamW(model.parameters())
    else:
        optimizer = ShardedOptimizer(model.parameters(), AdamW)

    model.train()
    total_times = []
    comm_times = []

    for _ in range(steps):
        x = torch.randint(0, 10000, (2, context_length), device=device)
        y = torch.randint(0, 10000, (2, 1), device=device)

        total_time, comm_time = run(
            model,
            optimizer,
            x,
            y,
            nsys,
        )

        total_times.append(total_time)
        comm_times.append(comm_time)

    mean_total_times = [
        torch.zeros(1, dtype=torch.float32, device=device)
        for _ in range(int(os.environ["WORLD_SIZE"]))
    ]
    mean_total_time = torch.tensor(
        np.mean(total_times).astype(np.float32), device=device
    )
    dist.all_gather(mean_total_times, mean_total_time)
    mean_comm_times = [
        torch.zeros(1, dtype=torch.float32, device=device)
        for _ in range(int(os.environ["WORLD_SIZE"]))
    ]
    mean_comm_time = torch.tensor(np.mean(comm_times).astype(np.float32), device=device)
    dist.all_gather(mean_comm_times, mean_comm_time)
    if int(os.environ["LOCAL_RANK"]) == 0:
        print(
            f"Average total time for forward and backward pass across all ranks: {mean_total_times}"
        )
        print(
            f"Average communication time for all-reduce across all ranks: {mean_comm_times}"
        )

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument("--shard_optimizer", action="store_true")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    main(
        args.context_length,
        args.model_size,
        args.steps,
        args.nsys,
        args.shard_optimizer,
    )
