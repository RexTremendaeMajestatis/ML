import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import config
import sys

# f = open('output.txt','w')
# sys.stdout = f

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(config.activation_function(self.conv1(x)))
        x = self.pool(config.activation_function(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = config.activation_function(self.fc1(x))
        x = config.activation_function(self.fc2(x))
        x = self.fc3(x)
        return x


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root=config.root_dir, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root=config.root_dir, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2)


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(net.parameters())
for epoch in range(config.epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % config.print_bound == config.print_bound - 1:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / config.print_bound:.3f}')
            running_loss = 0.0

torch.save(net.state_dict(), config.save_path)
