# %%
import math
from random import shuffle
import torch
from torch import nn
from torchvision import datasets, transforms
import torch.optim as optim

n = 1000

xs = list(range(1, 1000+1))
shuffle(xs)
# ys = xs.copy()
ys = [2*x + 5 for x in xs]

# %%

model = nn.Sequential(nn.Linear(1, 1))
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.003)


optimizer = optim.SGD(model.parameters(), lr=0.5)


epochs = 2000
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

tens = torch.Tensor([[5/n], [6/n], [700000/n]])
res = model(tens)
print(res)
res[1].item()

#  %%

print(model[0].weight)
print(model[0].bias)

# *2 * 1000 + 5

# %%

for x in xs:
    res = torch.Tensor([x]).to(torch.float32)
    val = res[0].item()
    assert not(math.isnan(val))
