import torch
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
#建立4个网络使用不同的优化器
SGD_net = Net()
Momentum_net = Net()
RMSprop_net = Net()
Adam_net = Net()
net_list = [SGD_net, Momentum_net, RMSprop_net, Adam_net]

opt_SGD = torch.optim.SGD(SGD_net.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(Momentum_net.parameters(), lr=LR, momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(RMSprop_net.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(Adam_net.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()

losses_his = [[], [], [], []]

lists = zip(net_list,optimizers,losses_his)
# 这个要是先zip会出问题

for epoch in range(EPOCH):
    print('第{}轮训练:'.format(epoch))
    for step, (batch_x, batch_y)  in enumerate(loader):
        '''
            每次训练拿32条数据，直到拿完
        '''
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        # print(step)
        # print(b_x)

        for net, opt, l_his in zip(net_list,optimizers,losses_his):
            predict = net(b_x)
            loss = loss_func(predict, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.data.item())


# 首先先看看12次训练的误差
for l_his in losses_his:
    print(l_his)

#画图

# labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
# for i, l_his in enumerate(losses_his):
#     plt.plot(l_his, label=labels[i])
# plt.legend(loc='best')
# plt.xlabel('Steps')
# plt.ylabel('Loss')
# plt.ylim((0, 0.2))
# plt.show()
