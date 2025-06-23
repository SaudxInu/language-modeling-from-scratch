import contextlib

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.cuda.nvtx as nvtx


class DDPIndividualParameters(nn.Module):
    def __init__(self, module: torch.nn.Module, nsys: bool = False) -> None:
        super().__init__()
        for param in module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    lambda param: self._post_accumulate_grad_hook(param)
                )
        self.module = module
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

    def _post_accumulate_grad_hook(self, param) -> None:
        with self.cm_nsys("backward_comm"):
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)
