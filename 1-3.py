#%%
import torch
torch.__version__

#%%
# lots of ways to create a tensor 
# https://pytorch.org/docs/stable/torch.html#creation-ops
# a tensor is really just a multidimensional array, starting
# with a simple empty array
e = torch.empty(2, 2)
e

#%%
# ok - that's strange -- there are values in that array! this
# isn't a true random, it's just whatever was in memory -- if 
# you really want random, which is pretty often
r = torch.rand(2, 2)
r

#%%
# that's more like it and sometimes you just want specific values
# like a good old zero
z = torch.zeros(2, 2)
z

#%%
# or specifc constants, let's make some threes!
c = torch.full((2, 2), 3)
c 

#%%
# the most flexible is the `torch.tensor` creation method, you can
# pass it data in a lot of formats -- starting with lists
l = torch.tensor([[1, 2], [3, 4]])
l

#%%
# as well as interoperate with numpy arrays, which is very
# handy to work with data you may have already processed
# with other machine learning tools liek sklearn
import numpy
n = numpy.linspace(0, 5, 5)
n

#%%
# turning this into pytorch is as easy as you would wish
nn = torch.tensor(n)
nn

#%%
# and back again is easy too!
nn.numpy()

#%%
# arrays support conventional operations -- size and slice
nn.shape

#%%
nn[1:], nn[0]

#%%
# in any creation method, you can also specify the data type
# like using a full precision floating point
s = torch.ones(3, 3, dtype=torch.float)
s

#%%
# all kinds of math operations are available
# https://pytorch.org/docs/stable/torch.html#math-operations
# math is straightforward operatos for common operations
# like addition
eye = torch.eye(3, 3)
eye + torch.zeros(3, 3)

#%%
# subtraction
eye - torch.ones(3, 3)

#%%
# broadcast multiplication of a constant
eye * 3

#%%
# or division...
eye / 3

#%%
# element wise tensor multiplication
eye * torch.full((3,3), 4)

#%%
# and you might not have seen this before, but a dot product
# operator in python
x = torch.rand(3, 4)
y = torch.rand(4, 3)
x @ y

#%%
# and handy machine learning component operations
# like getting the index of the maximum value
torch.tensor([1 , 2, 5, 3, 0]).argmax()