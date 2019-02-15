#%%
# Let's take a look at what is going on inside these convolutions
# by viewing the layer output channels as images. It's an interesting technique
# to get more of a feel for what the machine learner 'sees'


#%%%
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms

#%%
mnist = torchvision.datasets.MNIST('./var', download=True)
transform = transforms.Compose([transforms.ToTensor()])

train = torchvision.datasets.MNIST('./var', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
test = torchvision.datasets.MNIST('./var', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(test, batch_size=len(test), shuffle=True)

#%%
# let's plot a tensor as an image -- this hasn't had any machine learning
# just yet -- it is only the source image data

#%%
for inputs, outputs in trainloader:
    #slice out one channel
    image = inputs[0][0]
    plt.imshow(image.numpy(), cmap=plt.get_cmap('binary'))
    break



#%%
# OK -- that's an image - now let's train up a simple convolutional network
# and then augment it by saving intermediate tensors, the thing to know here
# is the convolutional tensors have multiple filters, we go from
# one color channel to three -- so we'll have some interesting choices when
# we visualize!



#%%
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
        self.after_conv1 = x
        x = self.pool(F.relu(self.conv2(x)))
        self.after_conv2 = x
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
# actuall does the softmax for us -- convience feature in pytorch

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
# ok -- now we have a trained model -- now we can visualize!
# pyplot is a bit odd when you make multiple images -- the 
# trick is to remember it is a bit modal - you create a figure
# which means the plots you call are 'to' that figure implicitly
# and then you add subplots which are (rows, columns,  index)
# and it is one based from left to right, top to bottom
#
# we'll make a figure with 3 rows, 6 columns to show the source
# image, then the first filter of three channels
# followed by the second filter of six channels


#%%
for inputs, outputs in trainloader:
    # multi image figure
    figure = plt.figure()
    # the original image
    image = inputs[0][0]
    
    figure.add_subplot(3, 6, 1)
    plt.imshow(image.numpy(), cmap=plt.get_cmap('binary'))
    output = net(inputs)
    # remember we have a batch in the model -- and this
    # has a gradient, so we'll need it detached to get numpy format
    filter_one = net.after_conv1[0].detach()
    for i in range(3):
        figure.add_subplot(3, 6, 6 + 1 + i)
        plt.imshow(filter_one[i].numpy(), cmap=plt.get_cmap('binary'))
    
    filter_two = net.after_conv2[0].detach()
    for i in range(6):
        figure.add_subplot(3, 6, 12 + 1 + i)
        plt.imshow(filter_two[i].numpy(), cmap=plt.get_cmap('binary'))
    plt.show()

    break
