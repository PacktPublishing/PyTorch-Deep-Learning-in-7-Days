#%%
# Images are quite convenient in pytorch, there are a few built
# in datasets -- let's take a look at the classic -- MNIST


#%%
import torch
import matplotlib.pyplot as plt
import numpy as np


#%%
import torchvision
mnist = torchvision.datasets.MNIST('./var', download=True)
mnist[0][0]

#%%
# looks like a squiggly 5 -- let's check the label
mnist[0][1]

#%%
# now the data is actually images, so we're going to need to 
# turn it into tensors, which is conveniently built in

#%%
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train = torchvision.datasets.MNIST('./var', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
test = torchvision.datasets.MNIST('./var', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(test, batch_size=len(test), shuffle=True)


#%%
# and now to define a very simple convolutional network

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # in channels, out channels (filters!), kernel size (square)
        # one channel -- this is greyscale
        self.conv1 = nn.Conv2d(1, 3, 3)
        # pooling divides in half with 
        # kernel size, stride the same as 2
        self.pool = nn.MaxPool2d(2, 2)
        # now here is where you start to need to think about
        # the size of the image
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.fc1 = nn.Linear(150, 128)
        self.fc2 = nn.Linear(128, 128)
        # ten digits -- ten outputs
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        # this is a good place to see the size for debugging
        # print(x.shape) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
#%%
# loss functions, here we are using cross entropy loss, which
# actuall does the softmax for us

#%%
import torch.optim as optim

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())


#%%
# and the training loop

#%%
for epoch in range(16):
    for inputs, outputs in trainloader:
        optimizer.zero_grad()
        results = net(inputs)
        loss = loss_function(results, outputs)
        loss.backward()
        optimizer.step()
    print("Loss: {0}".format(loss))

#%%
# now let's use that classification report to see how well we are doing


#%%
import sklearn.metrics
for inputs, actual in testloader:
    results = net(inputs).argmax(dim=1).numpy()
    accuracy = sklearn.metrics.accuracy_score(actual, results)
    print(accuracy)

print(sklearn.metrics.classification_report(actual, results))