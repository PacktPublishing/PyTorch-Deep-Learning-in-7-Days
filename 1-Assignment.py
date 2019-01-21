#%%
import torchvision
import torch

# get some data -- don't forget to download it
mnist = torchvision.datasets.MNIST('./var',
     download=True,
     transform=torchvision.transforms.ToTensor())

mnist[0]

#%%

# each batch, let's make a tensor of batch averages
batches = torch.utils.data.DataLoader(mnist, 
    batch_size=32)

batch_averages = torch.Tensor([
    batch[0].mean() for batch in batches
])

#%%
# and there we have it
batch_averages.mean()

#%% now just for kicks -- let's compute the average a bit by hand
# notice that the overall average is different than the batch-wise
# average -- this is normal something to think about when
# maching learning with batch training
all_images = torch.cat([
    image for image, label in mnist
])

all_images.shape, all_images.mean()