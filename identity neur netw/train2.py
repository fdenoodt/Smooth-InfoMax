# %%
import torch
from torch import nn
from torchvision import datasets, transforms
import torch.optim as optim

# Define a transform to normalize the data
# image = (image - mean) / std
# The parameters mean, std are passed as 0.5, 0.5 in your case. This will normalize the image in the range [-1,1].
# For example, the minimum value 0 will be converted to (0-0.5)/0.5=-1, the maximum value of 1 will be converted to (1-0.5)/0.5=1.
# if you would like to get your image back in [0,1] range, you could use,
# image = ((image * std) + mean)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST('~/.pytorch/MNIST_data/',
                          download=True, train=True, transform=transform)

type(trainset)
trainset
# %%

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)

# 784 -> 128 -> 64 -> 10
model = nn.Sequential(nn.Linear(784, 128),
                      # The rectified linear activation function or ReLU is a
                      # non-linear function or piecewise linear function that will output the
                      # input directly if it is positive, otherwise, it will output zero.
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))
# Define the loss
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

# %%


model

# images.shape = [32, 784]
for images, labels in trainloader:
    a = images
    b = images.view(images.shape[0], -1)
    print(a.shape)
    print(b.shape)

    print(labels.shape)
    break


  

# %%
images.shape

flattened_images = images.view(images.shape[0], -1)
flattened_images.shape

flattend_image = flattened_images[0]

# add 1 dimension: eg: [784] bekomes: [1, 784]
flattend_image = flattend_image.unsqueeze(0)

label = labels[0]
output = model(flattend_image)

print(label)
print(output)
# %%

X = torch.rand(1, 28 * 28, device='cpu')
print(X.shape)
model(X)
