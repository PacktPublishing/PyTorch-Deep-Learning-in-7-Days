#%%
# starting with imports
import torch

#%%
# now we'll create a model in the way that you are most likely
# going to use pytorch in practice, by creating a reusable model
# module -- this is simply a class with layer member variables
# and a forward method that does the actual computation
# we're going to build the same model we did before with linear
# and relu, but we'll also fix up our model and get the output
# shape we really want -- which is a tensor with two elements


#%%
# inputs and outputs, just random values -- we'll work with real 
# data in subsequent videos
inputs = torch.rand(1, 1, 64, 64)
outputs = torch.rand(1, 2)

#%%
# and now on to our module

class Model(torch.nn.Module):

    def __init__(self):
        '''
        The constructor is the place to set up each of the layers
        and activations.
        '''

        super().__init__()
        self.layer_one = torch.nn.Linear(64, 256)
        self.activation_one = torch.nn.ReLU()
        self.layer_two = torch.nn.Linear(256, 256)
        self.activation_two = torch.nn.ReLU()
        # this is a pretty big number -- because we are flattening
        # which turned 64 * 256 into a flat array like tensor
        self.shape_outputs = torch.nn.Linear(16384, 2)

    def forward(self, inputs):
        buffer = self.layer_one(inputs)
        buffer = self.activation_one(buffer)
        buffer = self.layer_two(buffer)
        buffer = self.activation_two(buffer)
        # and here -- we correct the model to give us the output 
        # shape we want -- starting with dimension one to 
        # preserve the batch dimension -- we only have a bactch
        # of one item, but dealing with batches will becomre more
        # important as we process real data in later videos
        buffer = buffer.flatten(start_dim=1)
        return self.shape_outputs(buffer)

#%%
# now let's run our model over our inputs

#%%
model = Model()
test_results = model(inputs)
test_results

#%%
# now -- let's learn, this time creating a learning loop
# with a built in optimizer -- we'll let it cycle keeping track
# of our gradients, and when our gradients 'vanish' -- we'll
# stop learning and see how close we are to our model
# being able to generate the correct outputs

#%%
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# a limit on how much learning we do
for i in range(10000):
    # the optimizer reaches into the model and will zero out
    optimizer.zero_grad()
    results = model(inputs)
    loss = loss_function(results, outputs)
    loss.backward()
    optimizer.step()
    # now -- look for vanishing gradients
    gradients = 0.0
    for parameter in model.parameters():
        gradients += parameter.grad.data.sum()
    if abs(gradients) <= 0.0001:
        print(gradients)
        print('gradient vanished at iteration {0}'.format(i))
        break


#%%
# relatively quick to get to no gradients, let's look at the answer
model(inputs), outputs

#%%
# spot on!
# this illustrates how networks can learn arbitrary functions,
# in this case -- extremely arbitrary, we learned random data!
# keep this in mind as you are doing machine learning on real data
# -- networks are powerful enough to fix nearly any data, including
# random, which means in effect the algorithm memorized the 
# inputs in a kind of sophisticated mathematical hashtable
# -- when this happens, we call it overfitting -- meaning
# the model knows only the inputs it is trained on and cannot
# deal with previously unseen inputs -- think about this 
# when you make your models!