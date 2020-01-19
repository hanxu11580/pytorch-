import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import numpy


EPOCH = 1
LR = 0.001
BATCH_SIZE = 50
DOWNLOAD_MNIST = False


train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# print(train_data.data.size())
# print(train_data.targets.size())

loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)


test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
)
test_x = Variable(torch.unsqueeze(test_data.data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255
test_y = test_data.targets[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,#图片为黑白，所以为1，RGB图片则为3
                out_channels=16,#提取16个特征(16个卷积核?)
                kernel_size=5, #窗口大小5*5
                stride=1,
                padding=2,#窗口若大于图片边缘了，将填0.   其中计算公式为：28+1=(28+2padding-kernel_size)/stride+1
            ), #过滤器， 用小窗口收集信息 ->(16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #->(16,14,14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), #->(32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) #->(32,7,7)
        )
        self.out = nn.Linear(32*7*7, 10) #10是10个数字

    def forward(self, x):
        x = self.conv1(x),
        x = torch.stack(x, dim=-1)
        x = self.conv2(x),
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    print('第{}轮训练:'.format(epoch))
    for step, (x, y) in enumerate(loader):
        b_x = Variable(x)
        b_y = Variable(y)

        predict = cnn(b_x)
        print('111')
        loss = loss_func(predict, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1).data.squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size(0)
            print('train loss: %.4f' % loss.item())
            print('accuracy:%.4f' % accuracy)