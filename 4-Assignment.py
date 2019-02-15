#%%
# Here I'm swapping out MNIST for CIFAR, which is object recognition
# -- and it is 3 channel color image


#%%%
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn



#%%
cifar = torchvision.datasets.CIFAR10('./var', download=True)
transform = transforms.Compose([
    transforms.ToTensor(),
])
cifar[0][0]

#%%
train = torchvision.datasets.CIFAR10('./var', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
test = torchvision.datasets.CIFAR10('./var', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(test, batch_size=len(test), shuffle=True)

#%%
class SlimAlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # three color input channels
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # this is the shape after flattening
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        # here is where I figure out the shape to get
        # the right number of parameters in the next layer
        # print(x.shape)
        x = self.classifier(x)
        return x


#%%
net = SlimAlexNet(num_classes=10)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

net.to(device)

# train this longer
for epoch in range(64):
    total_loss = 0
    for inputs, outputs in trainloader:
        inputs = inputs.to(device)
        outputs = outputs.to(device)
        optimizer.zero_grad()
        results = net(inputs)
        loss = loss_function(results, outputs)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print("Loss: {0}".format(total_loss / len(trainloader)))

#%%
# let's see how much better this is!

#%%
import sklearn.metrics
for inputs, actual in testloader:
    inputs = inputs.to(device)
    results = net(inputs).argmax(dim=1).to('cpu').numpy()
    accuracy = sklearn.metrics.accuracy_score(actual, results)
    print(accuracy)

print(sklearn.metrics.classification_report(actual, results))
