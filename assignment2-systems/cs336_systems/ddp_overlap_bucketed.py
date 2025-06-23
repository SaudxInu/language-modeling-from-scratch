from collections import defaultdict
from functools import partial
import contextlib

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.cuda.nvtx as nvtx


class DDPBucketedParameters(nn.Module):
    def __init__(
        self, module: torch.nn.Module, bucket_size_mb: float, nsys: bool = False
    ) -> None:
        super().__init__()
        self.buckets = defaultdict(list)
        bucket_index = 0
        bucket_size = 0.0
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.buckets[bucket_index].append((name, param))
                bucket_size = bucket_size + self._size_param(param)
                if bucket_size > bucket_size_mb:
                    bucket_index += 1
                    bucket_size = 0.0
            dist.broadcast(param.data, src=0)
        for bucket_index, bucket in self.buckets.items():
            bucket[0][1].register_post_accumulate_grad_hook(
                lambda param, bucket_index=bucket_index: self._post_accumulate_grad_hook(
                    param, bucket_index
                )
            )
        self.module = module
        self.grads_buffer = {}
        self.grads = {}
        self.cm_nsys = nvtx.range if nsys else contextlib.nullcontext
        self.handles = []

    def forward(self, *inputs, **kwargs):
        x = self.module(*inputs, **kwargs)
        return x

    def finish_gradient_synchronization(self):
        with self.cm_nsys("backward_comm"):
            for handle in self.handles:
                handle.wait()
        self.handles.clear()
        for bucket_index in self.grads_buffer:
            params = self.buckets[bucket_index]
            grads = torch._utils._unflatten_dense_tensors(
                self.grads_buffer[bucket_index], self.grads[bucket_index]
            )
            for param, grad in zip(params, grads):
                param[1].grad.copy_(grad)
        self.grads.clear()
        self.grads_buffer.clear()

    def _post_accumulate_grad_hook(self, param, bucket_index) -> None:
        params = self.buckets[bucket_index]
        grads = [
            (
                param[1].grad
                if param[1].grad is not None
                else torch.zeros(param[1].shape, device=f"cuda:{dist.get_rank()}")
            )
            for param in params
        ]
        grads_buffer = torch._utils._flatten_dense_tensors(grads)
        with self.cm_nsys("backward_comm"):
            handle = dist.all_reduce(grads_buffer, op=dist.ReduceOp.SUM, async_op=True)
        self.grads[bucket_index] = grads
        self.grads_buffer[bucket_index] = grads_buffer
        self.handles.append(handle)

    def _size_param(self, param: torch.nn.Parameter) -> float:
        return (param.numel() * param.element_size()) / (1024 * 1024)
