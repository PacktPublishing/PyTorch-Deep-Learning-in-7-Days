
#%%
# First, we'll need to load up a dataset. Pandas is a great
# tool to use to load csv data you may find, which we
# will later turn into tensors. 
# Let's start with the Dataset

#%%

import torch
import pandas
from torch.utils.data import Dataset

class MushroomDataset(Dataset):

    def __init__(self):
        '''Load up the data.
        '''
        self.data = pandas.read_csv('./mushrooms.csv')

    def __len__(self):
        '''How much data do we have?
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''Grab one data sample
        
        Arguments:
            idx {int} -- data at this position.
        '''
        return self.data.iloc[idx][0:1]
# pretty simple when we start from pandas
# here is a dataset loaded, with a single sample
shrooms = MushroomDataset()
len(shrooms), shrooms[0]

#%%
# Well -- we have some clearly identifiable properties, but we
# have this all in one dataset, we're going to need to separate
# out the inputs from the outputs

#%%
class MushroomDataset(Dataset):

    def __init__(self):
        '''Load up the data.
        '''
        self.data = pandas.read_csv('./mushrooms.csv')

    def __len__(self):
        '''How much data do we have?
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''Grab one data sample
        
        Arguments:
            idx {int, tensor} -- data at this position.
        '''
        # handle being passed a tensor as an index
        if type(idx) is torch.Tensor:
            idx = idx.item()
        return self.data.iloc[idx][1:], self.data.iloc[idx][0:1]

shrooms = MushroomDataset()
shrooms[0]

#%%
# One more thing to think about -- testing and training data
# we need some set of data samples we don't use in training to 
# verify that our model can generalize -- 
# that it can make a classification
# for an unseen sample and hasn't merely 
# memorized the input data

#%%
number_for_testing = int(len(shrooms) * 0.05)
number_for_training = len(shrooms) - number_for_testing
train, test = torch.utils.data.random_split(shrooms,
    [number_for_training, number_for_testing])
len(test), len(train)

#%%
test[0]
