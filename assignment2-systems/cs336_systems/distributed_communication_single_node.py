# torchrun --nnodes=1 --nproc_per_node=2 cs336_systems/distributed_communication_single_node.py
import os
import timeit

import torch
import torch.distributed as dist


DEVICE = "cuda"


def setup():
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup():
    dist.destroy_process_group()


def all_reduce():
    data = torch.randn((262_144 * 1,), device=DEVICE)
    dist.all_reduce(data, async_op=False)


def main():
    setup()
    for _ in range(5):
        dist.barrier()
        all_reduce()
        torch.cuda.synchronize()
    times = []
    for _ in range(10):
        dist.barrier()
        start_time = timeit.default_timer()
        all_reduce()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)
    avg_times = [
        torch.zeros(1, dtype=torch.float32, device=DEVICE)
        for _ in range(int(os.environ["WORLD_SIZE"]))
    ]
    avg_time = torch.tensor(sum(times) / len(times), device=DEVICE)
    dist.all_gather(avg_times, avg_time)
    if int(os.environ["LOCAL_RANK"]) == 0:
        print(f"Average time for all-reduce across all ranks: {avg_times}")
    cleanup()


main()
