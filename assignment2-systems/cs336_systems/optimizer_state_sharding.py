import os
from typing import Any, Type

from torch.optim import Optimizer
import torch.distributed as dist


class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        params = list(params)
        num_params = len(params)
        shard_size = num_params // world_size
        self.shards = {}
        for i in range(world_size):
            shard_start_idx = i * shard_size
            shard_end_idx = (
                shard_start_idx + shard_size if i < world_size - 1 else num_params
            )
            self.shards[i] = params[shard_start_idx:shard_end_idx]
        self.optimizer = optimizer_cls(self.shards[rank], **kwargs)

    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure=closure, **kwargs)
        for i, params in self.shards.items():
            for param in params:
                dist.broadcast(param.data, src=i)

    def add_param_group(self, param_group: dict[str, Any]):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        params = param_group["params"]
        params = list(params)
        num_params = len(params)
        shard_size = num_params // world_size
        shard_start_idx = rank * shard_size
        shard_end_idx = (
            shard_start_idx + shard_size if rank < world_size - 1 else num_params
        )
        param_group["params"] = params[shard_start_idx:shard_end_idx]
        self.optimizer.add_param_group(param_group)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
