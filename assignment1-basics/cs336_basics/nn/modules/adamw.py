from collections.abc import Callable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr,
        weight_decay,
        betas,
        eps,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]

                t = state.get("t", 1)

                m = state.get("m", 0.0)
                v = state.get("v", 0.0)

                m = (beta_1 * m) + ((1 - beta_1) * grad)
                v = (beta_2 * v) + ((1 - beta_2) * grad**2)

                alpha_t = lr * (math.sqrt(1 - beta_2**t) / (1 - beta_1**t))

                p.data -= (alpha_t * m) / (v**0.5 + eps)

                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1

                state["m"] = m
                state["v"] = v

        return loss
