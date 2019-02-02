#%%
# starting with imports
import torch

#%%
# here is some iteration -- lowering the number
# of hidden parameters until we no longer can get the
# gradiens to vanish
# this is a bit of dynamic model generation, which is 
# a kind of meta-learning


#%%
# inputs and outputs, just random values -- we'll work with real 
# data in subsequent videos
inputs = torch.rand(1, 1, 64, 64)
outputs = torch.rand(1, 2)

#%%
# keep track of how many learning steps it took
# at each number of parameters
learning_steps = []

#%%
for number_of_parameters in range(256, 1, -1):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_one = torch.nn.Linear(64, number_of_parameters)
            self.activation_one = torch.nn.ReLU()
            self.layer_two = torch.nn.Linear(number_of_parameters, number_of_parameters)
            self.activation_two = torch.nn.ReLU()
            # this is a pretty big number -- because we are flattening
            # which turned 64 * 256 into a flat array like tensor
            self.shape_outputs = torch.nn.Linear(number_of_parameters * 64, 2)

        def forward(self, inputs):
            buffer = self.layer_one(inputs)
            buffer = self.activation_one(buffer)
            buffer = self.layer_two(buffer)
            buffer = self.activation_two(buffer)
            buffer = buffer.flatten(start_dim=1)
            return self.shape_outputs(buffer)

    model = Model()
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
            learning_steps.append((number_of_parameters, i, results))
            break

#%%
learning_steps


#%%
# looks like it still learns -- but it gets a lot harder when
# the number of parameters converges to the number of outputs
import matplotlib.pyplot as plt
plt.style.use('ggplot')
learning_steps = [step[1] for step in learning_steps]
plt.plot(learning_steps)

