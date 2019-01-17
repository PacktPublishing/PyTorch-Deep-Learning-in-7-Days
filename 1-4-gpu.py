#%%
# You'll need a GPU to get this to run!
# `./linux/install-docker`
# `./linux/install-nvidia-docker`
# `./linux/run-nvidia-docker`
# get your notebook server started

# now let's talk about devices
import torch
cpu = torch.device('cpu')
gpu = torch.device('cuda')
cpu, gpu

#%%
# when you allocate a tensor, it's on a device, in the contexgt
# of that device, if you don't specify, it's on the CPU
x = torch.tensor([1.5])
x, x.device

#%%
# you can explicitly place, which is how I do it in general
y = torch.tensor([2.5], device=cpu)
y, y.device

#%%
# and now -- GPU
z = torch.tensor([3.5], device=gpu)
z, z.device

#%%
# you cannot mix devices, this is the important thing to remember
# particularly when loading up data -- make sure you put things
# together on a device!
x + y + z

#%% 
# but you can move things around to work on the gpu
a = x.to(gpu) + y.to(gpu) + z
a

#%%
# and you can move things back to the CPU
b = a.to(cpu)
b, b.device