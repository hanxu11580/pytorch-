import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy


EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False

train_data = dsets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_data = dsets.MNIST(
    root='./mnist',
    train=False,
    transform=torchvision.transforms.ToTensor()
)
test_x = Variable(test_data.data, volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.targets.numpy().squeeze()[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True #batch在第一个维度

        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None) #x.shape=(batch, time_step, input_size)
        out = self.out(r_out[:, -1, :]) #(batch,time_step,input_size) 选取最后一个时刻的output
        return out


rnn = RNN()
# print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    print('第1轮训练:')
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28, 28)) #相当于reshape， -1标识不确定
        b_y = Variable(y)

        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            sum = 0
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = numpy.sum(pred_y==test_y)/test_y.shape[0]
            print('step:{}'.format(step))
            print('loss:{}'.format(loss.item()))
            print('accuracy:{}'.format(accuracy))

