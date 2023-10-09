#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : my_torch_frame.py
# @Author: Ruiz Wu
# @Time  : 6/2/2021 3:59 PM
# @Desc  : torch test on dataset mnist
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter

from early_stopping import EarlyStopping


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)  # input channel, output channel, size
        self.conv2 = nn.Conv2d(16, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        in_size = x.size(0)  # batchsize or x.shape()[0]
        out = F.max_pool2d(F.relu(self.conv1(x)), 2)
        out = F.max_pool2d(F.relu(self.conv2(out)), 2)
        # print(out.shape)
        out = out.view(in_size, -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(out, dim=1)  # 计算log(softmax(x))
        return out


# example 1
# For reference
class ConvNet1(nn.Module):

    def __init__(self):
        super(ConvNet1, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# example 2
class ConvNet2(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.linear = nn.Linear(in_features=3 * 3 * 64, out_features=num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    writer = SummaryWriter('./tb_log')
    epoch_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)  # loss a batch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar(tag='loss', scalar_value=loss, global_step=batch_idx)
        epoch_loss.append(loss.item())
        if (batch_idx + 1) % 25 == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()), end='')
            sys.stdout.flush()

    return np.mean(epoch_loss)


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            # pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return test_loss


if __name__ == '__main__':
    # set seeds
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    # hype params
    BATCH_SIZE = 32
    EPOCHS = 3
    # set device
    print("GPU Available: " + str(torch.cuda.is_available()) + '\n' + 'torch.version: ' + torch.__version__)
    print('CUDA: ' + torch.version.cuda)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.set_default_tensor_type(torch.FloatTensor)
    # load data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=BATCH_SIZE, shuffle=True)
    # define model
    model = ConvNet2(num_classes=10).to(DEVICE)
    # model = resnet18().to(DEVICE)
    summary(model, input_size=(1, 28, 28))

    # train & test
    params = list(model.parameters())
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(save_path='\checkpoints')
    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, criterion, epoch)
        eval_loss = test(model, DEVICE, test_loader, criterion)

        # 早停止
        early_stopping(eval_loss, model)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练

    # model save & load
    # save
    # MODEL_PATH='./checkpoints/model_test.pt'
    # torch.save(model.state_dict(), MODEL_PATH)
    # load
    # model = ConvNet1()
    # model.load_state_dict(torch.load(MODEL_PATH))
