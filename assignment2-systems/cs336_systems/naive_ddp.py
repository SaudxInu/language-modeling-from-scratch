# torchrun --nnodes=1 --nproc_per_node=2 cs336_systems/naive_ddp.py
import os

import torch
import torch.distributed as dist
import torch.nn as nn

import torch
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # for reproducibility
    torch.backends.cudnn.benchmark = False  # slower but deterministic


def setup():
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 5, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    set_seed(0)
    setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ToyModel().to(device)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    print(
        f"Rank {dist.get_rank()} initialized model with parameters: {next(model.parameters())[0, :]}"
    )
    optmizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch_size = 10
    num_iterations = 5
    X = torch.randn(int(os.environ["WORLD_SIZE"]) * batch_size * num_iterations, 10)
    Y = torch.randn(int(os.environ["WORLD_SIZE"]) * batch_size * num_iterations, 5)
    start_idx = int(os.environ["LOCAL_RANK"]) * batch_size * num_iterations
    end_idx = start_idx + batch_size * num_iterations
    for _ in range(start_idx, end_idx, batch_size):
        x = X[start_idx:end_idx].to(device)
        y = Y[start_idx:end_idx].to(device)
        y_pred = model(x)
        loss = torch.nn.functional.mse_loss(y_pred, y)
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= int(os.environ["WORLD_SIZE"])
        optmizer.step()
        print(
            f"Rank {dist.get_rank()} updated model with parameters: {next(model.parameters())[0, :]}"
        )
    cleanup()


main()
