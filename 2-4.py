#%%
# Starting off, we need to import torch
import torch

#%%
# here are the inputs and outputs from the last video
# as well as the model with activations

#%%
inputs = torch.rand(1, 1, 64, 64)
outputs = torch.rand(1, 2)
model = torch.nn.Sequential(
    torch.nn.Linear(64, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 2),
)

#%%
# now we have our inputs as sample data and our outputs
# we are trying to generate with our network -- the 
# application of a network -- it's computation to turn input
# into output is done with a forward pass simply by calling
# the model as a function over the inputs

#%%
test_results = model(inputs)
test_results

#%%
# don't forget the loss function! the loss function is the
# key driver to compute gradients, which are needed to drive
# learning 

#%%
loss = torch.nn.MSELoss()(test_results, outputs)
loss


#%%
# now we compute the gradients, this is done for each forward
# pass after you have compute the loss -- basically you are
# zeroing out as the gradients will differ on each pass
# once the gradients are zeroed, then you use the loss to
# drive the backward propagation of gradients through the model
# this is pretty much the heart of what pytorch does for you -- 
# automatically computing gradients and propagating them back

#%%
model.zero_grad()
loss.backward()

#%%
# and now -- we'll do some very simple learning -- remember
# that the gradients tell you how far away you are from the
# right answer -- so you move in the opposite direction to
# get to the answer, meaning -- we just subtract!
# one additional concept here -- learning rate, which we'll 
# experiment with more in later videos, but for now we'll use
# a very simple constant learning rate

#%%
learning_rate = 0.001
for parameter in model.parameters():
    parameter.data -= parameter.grad.data * learning_rate


#%%
# now -- we've learned -- and should be closer to the answer
# let's run the model with our new updated parameters
# and see...

#%%
after_learning = model(inputs)
loss_after_learning = torch.nn.MSELoss()(after_learning, outputs)
loss_after_learning 

#%%
# yep -- that's a smaller loss, we are closer to the answer