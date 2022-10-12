# %%
import math
from random import shuffle
import torch
from torch import nn
import torch.optim as optim

n = 1000

xs = list(range(1, 1000+1))
shuffle(xs)
# ys = xs.copy()
ys = [2*x + 5 for x in xs]

# %%

model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1))
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.00004)


epochs = 8000
for e in range(epochs):
    xs_tensor = [[x/n] for x in xs]
    ys_tensor = [[y] for y in ys]

    xs_tensor = torch.Tensor(xs_tensor).to(torch.float32)
    ys_tensor = torch.tensor(ys_tensor).to(torch.float32)

    # Training pass
    optimizer.zero_grad()

    output = model(xs_tensor)

    loss = criterion(output, ys_tensor)
    loss.backward()
    optimizer.step()

    running_loss = loss.item()

    if e % 1000 == 0:
        print(f"Training loss: {running_loss}")


# %%


inp = [5, 6, 700000]
inp_preprocess = [[x/n] for x in inp]

tens = torch.Tensor(inp_preprocess)
res = model(tens)
for actual, result in zip(inp, res):
    print(actual, result.item())

#  %%

print(model)

inp = 6/n
res = inp
for layer in model:
    w = layer.weight.item()
    bias = layer.bias.item()

    res = (res * w) + bias

    print(f"W: {w} + b: {bias}")

print(f"inp: {inp*n}, out: {res}")

#Simplest:  (x * 2 * 1000) + 5

# %%

for x in xs:
    res = torch.Tensor([x]).to(torch.float32)
    val = res[0].item()
    assert not(math.isnan(val))
