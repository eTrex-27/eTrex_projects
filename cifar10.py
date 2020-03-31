import torch
import torch.nn as nn

import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import PIL.ImageOps
import requests
from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 20, 5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(20, 20, 5, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.fc = nn.Linear(4*4*20, 10)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = self.conv3(y)
        y = self.relu3(y)
        y = self.pool3(y)
        y = y.view(y.shape[0], -1)
        y = self.fc(y)
        return y

net = Net()
print(net)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
epochs = 20

if __name__ == '__main__':
    batch_size = 256
    data_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
    data_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))

    data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    data_test_loader = DataLoader(data_test, batch_size=batch_size)

    classes = ('plane', 'car', 'bird', 'cat', 'dear', 'dog', 'frog', 'horse', 'ship', 'truck')
    iters, losses = [], []
    loss_history = []
    correct_history = []
    n = 0
    for _epoch in range(epochs):
        loss = 0.0
        correct = 0.0
        for i, (images, labels) in enumerate(data_train_loader):
            optimizer.zero_grad()
            output = net(images)
            loss = loss_fn(output, labels)
            if i % 10 == 0:
                print('Epoch: {}, batch: {}, loss: {}'.format(_epoch, i, loss))
                print(net.fc.weight.detach().cpu().numpy().mean(axis=1))

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
        # epoch_loss = loss / len(data_train_loader)
        # epoch_acc = correct / len(data_train_loader)
        # loss_history.append(epoch_loss)
        # correct_history.append(epoch_acc)

    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    # plt.plot(loss_history, label='Training Loss')
    # plt.legend()
    # plt.show()
    # plt.plot(correct_history, label='Training accuracy')
    # plt.legend()
    # plt.show()