import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
'''
    torch只能处理二维数据
    torch.unsqueeze将一维数据变成二维数据
'''
# y = x.pow(2)
y = x.pow(2) + 0.2*torch.rand(x.size())
'''
    x^2 加入噪声点
    x.size()是[100,1]
    torch.rand(x.size())生成x.size()大小的二维数据
'''

x, y = Variable(x), Variable(y)
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        '''
            定义层
        :param n_features:
        :param n_hidden:
        :param n_output:
        '''
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    def forward(self, x):
        '''
            搭建层
        :return:
        '''
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(1, 10, 1)
# print(net) #可以查看网络结构

net1 = Net(2, 10, 2)

#方法2: 等价于上方的Class
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)
print(net1)
print(net2)


# plt.ion()
# plt.show()
#
# optimizer = torch.optim.SGD(net.parameters(), lr=1) #优化参数及设置学习效率
# SGD将数据分为小块进行传入训练
# loss_func = torch.nn.MSELoss() #均方差计算误差 MSE用于回归
# # loss_func = torch.nn.CrossEntropyLoss() #用于分类问题 输出的概率
#
# for t in range(100):
#     prediction = net(x)
#
#     loss = loss_func(prediction, y) #y是真实值
#
#     optimizer.zero_grad() #梯度设为0
#     loss.backward() #误差反向传递
#     optimizer.step() #优化梯度
#
#     if t % 5 == 0:
#         plt.cla()
#         plt.scatter(x.data.numpy(), y.data.numpy())
#         plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
#         plt.text(0.5, 0, 'Loss=%.5f' % loss.item(), fontdict={'size':20, 'color':'red'})
#         plt.pause(0.1)
#
# plt.ioff()
# plt.show()