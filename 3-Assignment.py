#%%
# Here is my version of the assignment working on graduate 
# school admission dataset as a regression problem

#%%
import torch
import pandas
import dateutil 
from torch.utils.data import Dataset

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


class DateEncoder():
    def __getitem__(self, datestring):
        '''Encode into a tensor [year, month, date]
        given an input date string.
        
        Arguments:
            datestring {string} -- date string, best bet is ISO format
        '''
        parsed = dateutil.parser.parse(datestring)
        return torch.Tensor([parsed.year, parsed.month, parsed.day])

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
#%%
categorical = [
    'Research',
]

dates = [

]

discard = [
    'Serial No.'
]
#%%

data = MixedCSV('./Admission_Predict.csv',
    'Chance of Admit ', #tricky follow space in the key name here!
    dates,
    categorical,
    discard
    )
#%%
# And here is a regression model
class Model(torch.nn.Module):

    def __init__(self, input_dimensions, size=256):
        '''
        The constructor is the place to set up each of the layers
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


# I ended up making a much smaller model, given the small number
# of features and small number of samples
model = Model(data[0][0].shape[0], size=32)
optimizer = torch.optim.Adam(model.parameters())
loss_function = torch.nn.MSELoss()


#%%
# and now our training loop
# I ended up with a much larger batch size

number_for_testing = int(len(data) * 0.05)
number_for_training = len(data) - number_for_testing
train, test = torch.utils.data.random_split(data,
    [number_for_training, number_for_testing])
training = torch.utils.data.DataLoader(train, batch_size=number_for_training, shuffle=True)
for epoch in range(256):
    for inputs, outputs in training:
        optimizer.zero_grad()
        results = model(inputs)
        loss = loss_function(results, outputs)
        loss.backward()
        optimizer.step()
    print("Loss: {0}".format(loss))

#%%
# quick check
actual = test[0][1]
predicted = model(test[0][0])
actual, predicted


#%%
import sklearn.metrics

testing = torch.utils.data.DataLoader(test, batch_size=len(test), shuffle=False)
for inputs, outputs in testing:
    predicted = model(inputs).detach().numpy()
    actual = outputs.numpy()
    print(sklearn.metrics.r2_score(actual, predicted))