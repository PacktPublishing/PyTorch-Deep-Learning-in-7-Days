#%%
# To do machine learning you need data, and there are three concepts
# to master here, Dataset, Dataloader, and transforms

#%%
# Let's make use of pandas and CSV data to create a dataset.

import torch
import pandas
from torch.utils.data import Dataset

class IrisDataset(Dataset):

    def __init__(self):
        '''Load up the data.
        '''
        self.data = pandas.read_csv('./Iris.csv')

    def __len__(self):
        '''How much data do we have?
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''Grab one data sample
        
        Arguments:
            idx {int} -- data at this position.
        '''
        return self.data.iloc[idx]
# pretty simple when we start from pandas
# here is a dataset loaded, with a single sample
iris = IrisDataset()
len(iris), iris[0]
#%%
# To do machine learning you need data, and there are three concepts
# to master here, Dataset, Dataloader, and transforms

#%%
# Let's make use of pandas and CSV data to create a dataset.

import torch
import pandas
from torch.utils.data import Dataset

class IrisDataset(Dataset):

    def __init__(self):
        '''Load up the data.
        '''
        self.data = pandas.read_csv('./Iris.csv')

    def __len__(self):
        '''How much data do we have?
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''Grab one data sample
        
        Arguments:
            idx {int} -- data at this position.
        '''
        return self.data.iloc[idx]
# pretty simple when we start from pandas
# here is a dataset loaded, with a single sample
iris = IrisDataset()
len(iris), iris[0]

#%%
# Now, the small problem is -- we have a named tuple,
# and we're going to need a tensor for inputs and
# the target label -- so we need to transform

class TensorIrisDataset(IrisDataset):
    def __getitem__(self, idx):
        '''Get a single sample that is 
        {values:, label:}
        '''
        sample = super().__getitem__(idx)
        return {
            'tensor': torch.Tensor(
                [sample.SepalLengthCm,
                sample.SepalWidthCm,
                sample.PetalLengthCm,
                sample.PetalWidthCm]
            ),
            'label': sample.Species
        }

# and output...
tensors = TensorIrisDataset()
len(tensors), tensors[0]

#%%
# Training almost always takes place in batches
# so pytorch has a very convenient loader that can take
# a dataset and turn it into batches so you can iterate
from torch.utils.data import DataLoader

loader = DataLoader(tensors, batch_size=16, shuffle=True)
for batch in loader:
    print(batch)

# see how the data comes out in batches, and the last batch
# tries to be as large as it can

#%%
# And -- there is even a parallel possibility 
# this is a pretty small dataset so it's not really
# essential, but here is how you use it

parallel_loader = DataLoader(tensors, 
    batch_size=16, shuffle=True, num_workers=4)
for batch in parallel_loader:
    print(batch)
