#%%
# Regression has a different output format than classification --
# it is in real values, good old floating point numbers.

# This particular dataset also has real valued data points, so we're
# going to update our data loader to be more configurable column
# wise how to encode values. We'll start be brining back our one
# hot column encoder.

#%%
import torch
import pandas

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

#%%
# now we could make an encoder for numerical values, but the values
# already are numbers, so this is in effect 'no encoder' -- so we'll
# implement it that way. 
# time to load up the dataset and learn about our columns

#%%
look = pandas.read_csv('./kc_house_data.csv')
look.iloc[0]


#%%
# looking at that data, let's make a configuration of the
# categorical columns, some of these are judgement -- for example
# it's easy to think of a 1-5 scale as categorical or as real valued, that
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
# you need to throw out, otherwise you can end up making a machine'
# learning hash-table!
# This isn't a hard and fast rule, but a good one to think about in practice
# any unique value in a sample isn't likely to generalize well, there isn't
# any data for the network to comapre. Whether it is a unique keyword
# or a unique number value, be on the lookout for these.

#%%
discard = [
    'id'
]

#%%
# And the really tricky bit -- look at that date column. Time is a tricky
# one to think about, as there are seasonal effects, and in some sense
# we recognize this in how we write out time -- years, months, and days
# let's break this feature into three numerical features.

#%%
import dateutil


class DateEncoder():
    def __getitem__(self, datestring):
        '''Encode into a tensor [year, month, date]
        given an input date string.
        
        Arguments:
            datestring {string} -- date string, best bet is ISO format
        '''
        parsed = dateutil.parser.parse(datestring)
        return torch.Tensor([parsed.year, parsed.month, parsed.day])

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
            categorical_series_names {list} -- column names of categories
            ignore_series_names {list} -- column names to skip
        '''
        self.dataset = pandas.read_csv(datafile)
        self.output_series_name = output_series_name
        self.encoders = {}
        for series_name in date_series_names:
            self.encoders[series_name] = DateEncoder()
        for series_name in categorical_series_names:
            self.encoders[series_name] = OneHotSeriesEncoder(
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