# embedding with eye to get one-hots? 
# wrapping transforming dataset?


#%%
# When we hae structured data, it is often cateogorical, meaning
# a series of discrete values, often strings or ID numbers, that
# are not intendted to be interpreted mathematically -- things
# as simple as category labels, states, country names -- or
# a numerical example -- postal codes are actually categorical

# In order to get this kind of data ready for machine learning, 
# we need to encode it properly -- which leads us to one hot encoding
# this is a method where each category is represented in a dataset as
# a single dimension, encoded 0 or 1 indicating if that category is set

# There are naturally a lot of ways to write the code to turn a
# categorical variable into a one hot encoded tensor -- one interesting
# idea is to use an identity matrix
# assuming you have three discreet values A, B, C, represent those
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

class OneHotSeriesEncoder():
    def __init__(self, series):
        '''Given a single pandas series, creaet an encoder
        that can turn values from that series into a one hot
        pytorch tensor.
        
        Arguments:
            series {pandas.Series} -- encode this
        '''
        unique_values = series.unique()
        self.ordinals = {val: i for i, val in enumerate(unique_values)}
        self.encoder = torch.eye(len(unique_values), len(unique_values))

    def __getitem__(self, value):
        '''Turn a value into a tensor
        
        Arguments:
            value {} -- Value to encode, anything that can be hashed
            but most likely a string
        
        Returns:
            [torch.Tensor] -- a one dimensional tensor with encoded values.
        '''

        return self.encoder[self.ordinals[value]]

class CategoricalCSV(Dataset):
    def __init__(self, datafile, output_series_name):
        '''Load the dataset and create needed encoders for
        each series.
        
        Arguments:
            datafile {string} -- path to data file
            output_series_name {string} -- use this series/column as output
        '''
        self.dataset = pandas.read_csv(datafile)
        self.output_series_name = output_series_name
        self.encoders = {}
        for series_name, series in self.dataset.items():
            # create a per series encoder
            self.encoders[series_name] = OneHotSeriesEncoder(series)

    def __getitem__(self, index):
        '''Return an (input, output) tensor tuple
        with all categories one hot encoded.
        
        Arguments:
            index {[type]} -- [description]
        '''
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

#%% 2-3