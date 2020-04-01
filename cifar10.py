import torch
import torch.nn as nn

import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 30, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(30)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(30, 30, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(30)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(30, 80, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(80)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(80, 80, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(80)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(80, 160, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(160)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(160, 160, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(160)
        self.relu6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4 * 4 * 160, 4096)
        self.relu7 = nn.ReLU()
        self.fc2 = nn.Linear(4096, 1000)
        self.relu8 = nn.ReLU()
        self.fc3 = nn.Linear(1000, 10)
        self.relu9 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.pool1(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu3(y)
        y = self.conv4(y)
        y = self.bn4(y)
        y = self.relu4(y)
        y = self.pool2(y)
        y = self.conv5(y)
        y = self.bn5(y)
        y = self.relu5(y)
        y = self.conv6(y)
        y = self.bn6(y)
        y = self.relu6(y)
        y = self.pool3(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu7(y)
        y = self.fc2(y)
        y = self.relu8(y)
        y = self.fc3(y)
        return y

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Net().to(device)
print(net)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
epochs = 20

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

if __name__ == '__main__':
    batch_size = 256
    data_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    data_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    data_test_loader = DataLoader(data_test, batch_size=batch_size)

    classes = ('plane', 'car', 'bird', 'cat', 'dear', 'dog', 'frog', 'horse', 'ship', 'truck')
    loss_train_history, correct_train_history, loss_test_history, correct_test_history = [], [], [], []
    iters, iters2, losses_train, accuracies_train, losses_test, accuracies_test = [], [], [], [], [], []
    n, k = 0, 0
    for _epoch in range(epochs):
        train_loss = 0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(data_train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            iters.append(n)
            losses_train.append(loss / batch_size)
            n += 1

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            accuracies_train.append(100. * correct / total)
        print('Epoch: %d | Loss_train: %.3f | Acc_train: %.3f%% (%d/%d)' % (_epoch, train_loss / len(data_train_loader), 100. * correct / total, correct, total))

        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_test_loader):
                images, labels = images.to(device), labels.to(device)
                output = net(images)
                loss = loss_fn(output, labels)

                iters2.append(k)
                losses_test.append(loss / batch_size)
                k += 1

                test_loss += loss.item()
                _, predicted = output.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                accuracies_test.append(100. * correct / total)
            print('Epoch: %d | Loss_test: %.3f | Acc_test: %.3f%% (%d/%d)' % (_epoch, test_loss / (i + 1), 100. * correct / total, correct, total))

    plt.plot(iters, losses_train, label='Training Loss')
    plt.legend()
    plt.show()
    plt.plot(iters, accuracies_train, label='Training accuracy')
    plt.legend()
    plt.show()
    plt.plot(iters2, losses_test, label='Test Loss')
    plt.legend()
    plt.show()
    plt.plot(iters2, accuracies_test, label='Test accuracy')
    plt.legend()
    plt.show()
