#%%
# there are also a lot of predefined datasets in torchvision
import torch
import torchvision
import matplotlib.pyplot as plt
dir(torchvision.datasets)

#%%
# let's take a look at some of this data
# really handy built in download!
cifar = torchvision.datasets.CIFAR10('./var', download=True)
cifar[0]

#%%
# looks like this is an image
fig = plt.figure(figsize=(1,1))
sub = fig.add_subplot(111)
sub.imshow(cifar[0][0])

#%% 
# how, that's a frog -- but -- we need a tensor of a
# frog -- so that's where transforms come in
# transforms are built in to torchvision and are
# objects that implement __call__ can change the data
from torchvision import transforms
pipeline = transforms.Compose([
    transforms.ToTensor()
    ])
cifar_tr = torchvision.datasets.CIFAR10('./var', transform=pipeline)

#%%
cifar_tr[0]