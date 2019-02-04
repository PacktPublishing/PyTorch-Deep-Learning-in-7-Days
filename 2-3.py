#%%
# Starting off, we need to import torch

#%%
import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#%%
# here are the inputs and outputs from the last videoA

#%%
inputs = torch.rand(1, 1, 64, 64)
outputs = torch.rand(1, 2)

#%%
# here is our model from the last video -- notice everything is 
# linear -- this limits what out model can learn, so we need
# something else -- an activation function

#%%
model = torch.nn.Sequential(
    torch.nn.Linear(64, 256),
    torch.nn.Linear(256, 256),
    torch.nn.Linear(256, 2),
)

#%%
# this is just about the simplest activation functional possible
# the RELU
# it is nonlinear in a straightforward way -- it starts out flat
# and then it inflects at 0 -- two linear parts making
# non linear part -- the advantage is -- it is very fast

#%%
x = torch.range(-1, 1, 0.1)
y = torch.nn.functional.relu(x)
plt.plot(x.numpy(), y.numpy())

#%%
# now we can update our model with relu
model = torch.nn.Sequential(
    torch.nn.Linear(64, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 2),
)

#%%
# and the model needs feedback in order to learn, and this is the
# role of the loss function -- it simply tells you how far away
# from the right answer we are
# you can think of -- and it's not to far off -- of machine
# learning as taking a random guess, and saying 'how far wrong'
# and then updating that guess -- this would be a silly strategy
# as a person, but computers can guess fast
#
# there are a lot of choices for loss functions, a classic
# one is the Mean Squared Error, this is related to the 
# classic distance you may have learned in school --
# A^2 + B^2 = C^2 when looking at right triangles, but
# -- this is generalized into high dimension tensors
# you'll hear it referred to as the L2 (because of the square)
# or Euclidean distance as well

#%%
results = model(inputs)
loss = torch.nn.MSELoss()(results, outputs)
loss

#%%
# and finally -- the gradient -- when I said machine learning
# makes a lot of guesses, there is a bit more to it - it
# makes educated guesses -- that education is in the gradient
#
# the gradient tells the machine learning model, based on 
# the loss -- which direction it is away from the right 
# answer
# and that will be the subject of our next video