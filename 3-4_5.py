#%%
# Regression has a different output format than classification --
# it is in real values, good old floating point numbers.

# This particular dataset also has real valued data points, 
# so we're
# going to update our data loader to be more configurable column
# wise how to encode values. We'll start be brining back our one
# hot column encoder.

#%%
import torch
import pandas

class OneHotEncoder():
    def __init__(self, series):
        '''Given a single pandas series, create an encoder
        that can turn values from that series into a one hot
        pytorch tensor.
        
        Arguments:
            series {pandas.Series} -- encode this
        '''
        unique_values = series.unique()
        self.ordinals = {
            val: i for i, val in enumerate(unique_values)}
        self.encoder = torch.eye(
            len(unique_values), len(unique_values))

    def __getitem__(self, value):
        '''Turn a value into a tensor
        
        Arguments:
            value {} -- Value to encode
            but most likely a string
        
        Returns:
            [torch.Tensor] -- a one dimensional tensor 
        '''

        return self.encoder[self.ordinals[value]]

#%%
# now we could make an encoder for numerical values, 
# but the values
# already are numbers, so this is in effect 'no encoder' -- i
# so we'll implement it that way. 
# time to load up the dataset and learn about our columns

#%%
look = pandas.read_csv('./kc_house_data.csv')
look.iloc[0]


#%%
# looking at that data, let's make a configuration of the
# categorical columns, some of these are judgement -- for example
# it's easy to think of a 1-5 scale as categorical or 
# as real valued, that
# will be something to experiment with in the assignment

#%%
categorical = [
    'waterfront',
    'view',
    'condition',
    'grade',
]

#%%
# One interesting column in there -- id -- that one doesn't look like real 
# data, just a key from a database. Values that are 'unique' like this
# you need to throw out, otherwise you can end up making a 
# machine learning hash-table!
# This isn't a hard and fast rule, but a good one to think
# about in practice
# any unique value in a sample isn't likely to generalize well,
# there isn't
# any data for the network to comapre. 
# Whether it is a unique keyword
# or a unique number value, be on the lookout for these.

#%%
discard = [
    'id'
]

#%%
# And the really tricky bit -- look at that date column. 
# Time is a tricky
# one to think about, as there are seasonal effects, 
# and in some sense
# we recognize this in how we write out time -- 
# years, months, and days
# let's break this feature into three numerical features.

#%%
import dateutil


class DateEncoder():
    def __getitem__(self, datestring):
        '''Encode into a tensor [year, month, date]
        given an input date string.
        
        Arguments:
            datestring {string} -- date string, ISO format
        '''
        parsed = dateutil.parser.parse(datestring)
        return torch.Tensor(
            [parsed.year, parsed.month, parsed.day])

dates = ['date']
DateEncoder()['20141013T000000']


#%%
from torch.utils.data import Dataset

class MixedCSV(Dataset):
    def __init__(self, datafile, output_series_name,
        date_series_names, categorical_series_names,
        ignore_series_names):
        '''Load the dataset and create needed encoders for
        each series.
        
        Arguments:
            datafile {string} -- path to data file
            output_series_name {string} -- use this series/column as output
            date_series_names {list} -- column names of dates
            categorical_series_names {list} -- column names
            ignore_series_names {list} -- column names to skip
        '''
        self.dataset = pandas.read_csv(datafile)
        self.output_series_name = output_series_name
        self.encoders = {}
        for series_name in date_series_names:
            self.encoders[series_name] = DateEncoder()
        for series_name in categorical_series_names:
            self.encoders[series_name] = OneHotEncoder(
                self.dataset[series_name]
            )
        self.ignore = ignore_series_names

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

        output = torch.Tensor([sample[self.output_series_name]]) 

        input_components = []
        for name, value in sample.items():
            if name in self.ignore:
                continue
            elif name in self.encoders:
                input_components.append(
                    self.encoders[name][value]
                )
            else:
                input_components.append(torch.Tensor([value]))
        input = torch.cat(input_components)
        return input, output

houses = MixedCSV('./kc_house_data.csv',
    'price',
    dates,
    categorical,
    discard
    )
houses[0]





#%%
# 3-5
# The big differences for a regression network are in the output
# rather than a softmax or a probability, 
# we can simply emit a real valued # number
# Depending on the model - 
# this number isn't simply a 0-1, and in our case
# we're looking to emit a price in dollars, 
# so it'll be 6 figures.

#%%
class Model(torch.nn.Module):

    def __init__(self, input_dimensions, size=128):
        '''
        The constructor is the place to set up each layer
        and activations.
        '''
        super().__init__()
        self.layer_one = torch.nn.Linear(input_dimensions, size)
        self.activation_one = torch.nn.ReLU()
        self.layer_two = torch.nn.Linear(size, size)
        self.activation_two = torch.nn.ReLU()
        self.shape_outputs = torch.nn.Linear(size, 1)

    def forward(self, inputs):

        buffer = self.layer_one(inputs)
        buffer = self.activation_one(buffer)
        buffer = self.layer_two(buffer)
        buffer = self.activation_two(buffer)
        buffer = self.shape_outputs(buffer)
        return buffer

model = Model(houses[0][0].shape[0])
optimizer = torch.optim.Adam(model.parameters())
loss_function = torch.nn.MSELoss()


#%%
# and now our training loop

number_for_testing = int(len(houses) * 0.05)
number_for_training = len(houses) - number_for_testing
train, test = torch.utils.data.random_split(houses,
    [number_for_training, number_for_testing])
training = torch.utils.data.DataLoader(
    train, batch_size=64, shuffle=True)
for epoch in range(16):
    for inputs, outputs in training:
        optimizer.zero_grad()
        results = model(inputs)
        loss = loss_function(results, outputs)
        loss.backward()
        optimizer.step()
    print("Loss: {0}".format(loss))


#%%
# notice those loss numbers are large, since we are computing
# loss in terms of dollars, and the error involves a square, 
# you always
# need to get a sense of error relative to your target numbers 
# -- another
# way to think of this would be to create a model 
# that gives an output
# on the range 0-1 and multiply that by the 
# range of values you
# see in the output say 0-1000000 for houses, 
# but as you can see from these
# outputs, our model seems plenty well able to 
# learn with large number output

# let's use our test data and see what we get

#%%
# here is our actual , and w
actual = test[0][1]
predicted = model(test[0][0])
actual, predicted

#%%
# wow - that's pretty good for an quick eyeball check, let's 
# take a look at the overall error for all our test data
# for this we'll reach back to sklearn and use R^2, 
# which gives a score
# of 0-1, one being best, and is a standard method 
# to judge the quality
# of a regression model

#%%
import sklearn.metrics
import torch.utils.data

testing = torch.utils.data.DataLoader(
    test, batch_size=len(test), shuffle=False)
for inputs, outputs in testing:
    predicted = model(inputs).detach().numpy()
    actual = outputs.numpy()
    print(sklearn.metrics.r2_score(actual, predicted))


#%%
# pretty good, that's actually better than I expected 
# when I started 
# up this model -- turns out house prices are quite predictable
# in this dataset -- I've actually seen this done 
# for local housing prices
# by some of my co workers when they were moving 
# to maximize their
# return and minimze their risk -- 
# a pretty useful application if you 
# can get some local MLS data and are planning a move!
