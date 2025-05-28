import numpy as np


def learning_rate_schedule(t, alpha_max, alpha_min, t_warmup, t_decay):
    if t < t_warmup:
        return alpha_max * (t / t_warmup)
    elif t < t_decay:
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (
            1 + np.cos(np.pi * (t - t_warmup) / (t_decay - t_warmup))
        )
    else:
        return alpha_min
