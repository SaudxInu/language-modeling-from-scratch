import numpy as np
import torch


def data_loading(x, batch_size, context_length, device):
    n = x.shape[0]

    start_idx = np.random.randint(
        0, high=n - context_length, size=batch_size, dtype=int
    )

    x_batch = np.array([x[s : s + context_length] for s in start_idx])
    y_batch = np.array([x[s + 1 : s + context_length + 1] for s in start_idx])

    x_batch = torch.tensor(x_batch, device=device, dtype=torch.int16)
    y_batch = torch.tensor(y_batch, device=device, dtype=torch.int16)

    return x_batch, y_batch
