#%%
# Starting off, we need to import torch
import torch

#%%
# this neural network will focus just on creating the network
# and not on data or data loading, so we'll keep this simple and
# build up the network step by step, in this first set of code
# we'll work with dummy random inputs and outputs, and connect
# them in a network

#%%
# inputs - -this is a 'batch' of size 1, with 1 color channel --
# imagine this is greyscale, and 64x and 64y pixels

#%%
inputs = torch.rand(1, 1, 64, 64)
inputs

#%%
# outputs -- pretend we are building a binary classifier,
# so we'll have to output possibilites, with a batch size of 1
# we'll use rand again, so each thing can be a little bit
# category 0 and a little bit category 1

#%%
outputs = torch.rand(1, 2)
outputs

#%%
# OK in a real model, those inputs and outputs would be your actual
# data, loaded up, in datasets, converted into batches
# for this simple model, it's just tensors, no data wrangling 
# now for a sequential network, let's do a simple multi
# layer perceptron


#%%
# now we start up a model with layers of linear -- these will
# themselves have tensors inside filled with random numbers
# these random numbers are called parameters, and these 
# parameters are the things that machine learning learns
# basically -- the parameters -- sometimes called weights
# are updated by learning algorithms, searching for the best
# available answer

#%%
model = torch.nn.Sequential(
    # input features are the size of one image
    # outputs are how many we have when done
    # the 64 has to 'match' the final dimnension of the input
    # try changing it to another number to see errors!
    torch.nn.Linear(64, 256),
    torch.nn.Linear(256, 256),
    torch.nn.Linear(256, 2),
)

#%%
# and -- this isn't learning, we're just running our random
# initialized linear network over our input

#%%
result = model(inputs)
result, result.shape

#%%
# hmm -- that's not two convenient output labels, we have some
# more work to do in the next videos -- but we have a model!
