import torch


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

s = torch.tensor(0, dtype=torch.float32, device=device)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32, device=device)
print(s)

s = torch.tensor(0, dtype=torch.float16, device=device)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16, device=device)
print(s)

s = torch.tensor(0, dtype=torch.float32, device=device)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16, device=device)
print(s)

s = torch.tensor(0, dtype=torch.float32, device=device)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16, device=device)
    s += x.type(torch.float32)
print(s)
