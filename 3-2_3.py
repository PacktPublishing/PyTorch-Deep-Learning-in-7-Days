

#%%
# When we have structured data, it is often cateogorical, meaning
# a series of discrete values, often strings or ID numbers, that
# are not intendted to be interpreted mathematically -- things
# as simple as category labels, states, country names -- or
# a numerical example -- postal codes are actually categorical

# In order to get this kind of data ready for machine learning, 
# we need to encode it properly -- which leads us to 
# one hot encoding
# this is a method where each category is represented 
# in a dataset as a single dimension, 
# encoded 0 or 1 indicating if that category is set

# There are naturally a lot of ways to write the code to turn a
# categorical variable into a one hot encoded tensor -- 
# one interesting idea is to use an identity matrix
# assuming you have three discreet values A, B, C, 
# represent those
# as a list ['A', 'B', 'C'], and then A is 0, B is 1, C is 2 as
# ordinal index values -- we can use an identity matrix to turn
# the ordinal into a one hot encoded representation

#%%
import torch
from torch.utils.data import Dataset
import pandas

#%%
one_hots = torch.eye(3, 3)
one_hots

#%%
ordinals = {c: i for i, c in enumerate(['A', 'B', 'C'])}
ordinals

#%%
one_hots[ordinals['A']]

#%%
# so this is an encoder - turning the letters into a bitmap
# now we need to just generalize this to work on a whole dataset!

#%%

class OneHotEncoder():
    def __init__(self, series):
        '''Given a single pandas series, creaet an encoder
        that can turn values from that series into a one hot
        pytorch tensor.
        
        Arguments:
            series {pandas.Series} -- encode this
        '''
        unique_values = series.unique()
        self.ordinals = {
            val: i for i, val in enumerate(unique_values)
            }
        self.encoder = torch.eye(
            len(unique_values), len(unique_values)
            )

    def __getitem__(self, value):
        '''Turn a value into a tensor
        
        Arguments:
            value {} -- Value to encode, 
            anything that can be hashed but most likely a string
        
        Returns:
            [torch.Tensor] -- a one dimensional tensor
        '''

        return self.encoder[self.ordinals[value]]

class CategoricalCSV(Dataset):
    def __init__(self, datafile, output_series_name):
        '''Load the dataset and create needed encoders for
        each series.
        
        Arguments:
            datafile {string} -- path to data file
            output_series_name {string} -- series/column name
        '''
        self.dataset = pandas.read_csv(datafile)
        self.output_series_name = output_series_name
        self.encoders = {}
        for series_name, series in self.dataset.items():
            # create a per series encoder
            self.encoders[series_name] = OneHotEncoder(series)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        '''Return an (input, output) tensor tuple
        with all categories one hot encoded.
        
        Arguments:
            index {[type]} -- [description]
        '''
        if type(index) is torch.Tensor:
            index = index.item()
        sample = self.dataset.iloc[index]
        output = self.encoders[self.output_series_name][
            sample[self.output_series_name]
        ]
        input_components = []
        for name, value in sample.items():
            if name != self.output_series_name:
                input_components.append(
                    self.encoders[name][value]
                )
        input = torch.cat(input_components)
        return input, output


shrooms = CategoricalCSV('./mushrooms.csv', 'class')
shrooms[0]

#%%
# now that is a lot of ones and zeros, but if you look closer, you can see
# that out of each range of numbers, there is a clearly a 1 signal
# that category value flag is on

# at this point -- we have tensors from pure CSC text data -- and 
# are ready to start learning!

#%% 3-3
# let's start off creating a simple network, we'll make the
# input and output dimensions variable, and introduce a new
# activation -- Softmax -- this is a function that smooths out
# values to get the total of all possibilities to sum to 1
# you may recognize this as a probability -- that's the effect
# you can take classifier results and nudge them into probability
# classes, which is very useful with a binary classifier trying
# to distinguish one thing from another

# a new loss function -- cross entropy -- makes a debut, 
# which is # very useful for classifiers, 
# it differs from mean squared error
# in that it is less concerned about the actual value 
# in a prediction
# than it is in which prediction has the largest value -- 
# the idea here
# is that for a binary classifier, the answer with the highest 
# probability is 'the' prediction, even though it may not be 100%

#%%
class Model(torch.nn.Module):

    def __init__(self, input_dimensions, 
        output_dimensions, size=128):
        '''
        The constructor is the place to set up each of the layers
        and activations.
        '''
        super().__init__()
        self.layer_one = torch.nn.Linear(input_dimensions, size)
        self.activation_one = torch.nn.ReLU()
        self.layer_two = torch.nn.Linear(size, size)
        self.activation_two = torch.nn.ReLU()
        self.shape_outputs = torch.nn.Linear(size, 
            output_dimensions)

    def forward(self, inputs):

        buffer = self.layer_one(inputs)
        buffer = self.activation_one(buffer)
        buffer = self.layer_two(buffer)
        buffer = self.activation_two(buffer)
        buffer = self.shape_outputs(buffer)
        return torch.nn.functional.softmax(buffer, dim=-1)

model = Model(shrooms[0][0].shape[0], shrooms[0][1].shape[0])
optimizer = torch.optim.Adam(model.parameters())
loss_function = torch.nn.BCELoss()

#%%
# now let's run a training loop, we'll go through the dataset
# multiple times -- a loop through the dataset is conventionally
# called an epoch, inside of each epoch, 
# we'll go through each batch

#%%
number_for_testing = int(len(shrooms) * 0.05)
number_for_training = len(shrooms) - number_for_testing
train, test = torch.utils.data.random_split(shrooms,
    [number_for_training, number_for_testing])
training = torch.utils.data.DataLoader(train, 
    batch_size=16, shuffle=True)
for epoch in range(4):
    for inputs, outputs in training:
        optimizer.zero_grad()
        results = model(inputs)
        loss = loss_function(results, outputs)
        loss.backward()
        optimizer.step()
    print("Loss: {0}".format(loss))


#%%
# now let's take a look at accuracy, this is a place where 
# we can reach to sklearn - 
# which has multiple evaluation functions and metrics

#%%
import sklearn.metrics

#%%
# accuracy is somewhat what you'd guess,
# the percentage of the time
# that your model is getting the right answer -- 
# let's look with
# our test data, 
# comparing the actual test output with our model 
# output on test

# here we are taking the argmax -- this turns one-hots back into
# integers -- say you have a one hot [1, 0] -- the arg max is 0 
# since the maximum value is in slot zero

# we compute the argmax along dimension 1 -- 
# dim=1, remember dim 0
# in this case is the batch, so each batch entry is indexed in
# 0 dimension, each one hot encoding is in the 1 dimension

#%%
testing = torch.utils.data.DataLoader(test, 
    batch_size=len(test), shuffle=False)
for inputs, outputs in testing:
    results = model(inputs).argmax(dim=1).numpy()
    actual = outputs.argmax(dim=1).numpy()
    accuracy = sklearn.metrics.accuracy_score(actual, results)
    print(accuracy)

#%%
# and, you can see how accurate you are -- per class this is
# a way to tell if you model is better or worse at making i
# certain
# kinds of predictions

#%%
sklearn.metrics.confusion_matrix(actual, results)

#%%
# you read this left to right
# true positive, false positive
# false negative, true negative

#%%
# even better, you can get a handy classification report, 
# which is easy to read

#%%
print(sklearn.metrics.classification_report(actual, results))
