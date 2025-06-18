import torch
import torch.nn as nn

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        print("fc1 dtype:", x.dtype)
        x = self.relu(x)
        print("relu dtype:", x.dtype)
        x = self.ln(x)
        print("ln dtype:", x.dtype)
        x = self.fc2(x)
        print("fc2 dtype:", x.dtype)
        return x


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = ToyModel(100, 10)
model.to(device)

optimizer = AdamW(model.parameters())

x = torch.randn(2, 100, dtype=torch.float32, device=device)
y = torch.randint(0, 10, (2,), device=device)

with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    for name, p in model.named_parameters():
        print(name, p.dtype)

    y_pred = model(x)

    loss = cross_entropy(y_pred, y)
    print("loss dtype:", loss.dtype)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    for name, p in model.named_parameters():
        print(name, p.grad.data.dtype)
