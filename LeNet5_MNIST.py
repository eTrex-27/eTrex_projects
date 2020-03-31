import torch
import torch.nn as nn

import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = self.conv3(y)
        y = self.relu3(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu4(y)
        y = self.fc2(y)
        return y

net = Net()
print(net)

if __name__ == '__main__':
    batch_size = 256
    print("Load files...")
    data_train = MNIST('./data/mnist', train = True, download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
    data_test = MNIST('./data/mnist', train=False, download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))

    data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    data_test_loader = DataLoader(data_test, batch_size=batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    epoch = 100

    loss_list, batch_list = [], []
    for _epoch in range(epoch):  
        for i, (images, labels) in enumerate(data_train_loader):
            optimizer.zero_grad()
            output = net(images)
            loss = loss_fn(output, labels)
            if i % 10 == 0:
                print('batch: {}, loss: {}'.format(i, loss))

            loss.backward()
            optimizer.step()
            
            loss.backward()
            optimizer.step()
            iters.append(n)
            losses.append(loss / batch_size)
            n += 1

        total_correct = 0
        avg_loss = 0.0
        for i, (images, labels) in enumerate(data_test_loader):
            output = net(images)
            avg_loss += loss_fn(output, labels).sum()
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            avg_loss /= len(data_test)
        print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))
        
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
