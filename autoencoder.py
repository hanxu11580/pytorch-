import torch
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

torch.manual_seed(1)

EPOCH = 10
BATCH_SIZE = 64
LR = 0.05
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

train_data = torchvision.datasets.MNIST(
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

# print(train_data.data.size()) # [60000, 28, 28]
# plt.imshow(train_data.data[2].numpy(), cmap='gray')
# plt.title('%i' % train_data.targets[2].numpy())
# plt.show()


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28*28, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 12),
            torch.nn.Tanh(),
            torch.nn.Linear(12, 3)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 12),
            torch.nn.Tanh(),
            torch.nn.Linear(12, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 28*28),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = torch.nn.MSELoss()

# f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
# plt.ion()
# plt.show()
#
# view_data = Variable(train_data.data[:N_TEST_IMG]).view(-1, 28*28).type(torch.FloatTensor)
# for i in range(N_TEST_IMG):
#     a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
#     a[0][i].set_xticks(())
#     a[0][i].set_yticks(())


for epoch in range(EPOCH):
    print('第{}轮'.format(epoch))
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28*28))
        b_y = Variable(x.view(-1, 28*28))
        b_label = Variable(y)

        encoded, decoded = autoencoder(b_x)
        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#         if step % 100 == 0:
#             print('loss:{}'.format(loss.item()))
#
#             _,decoded_data = autoencoder(view_data)
#             for i in range(N_TEST_IMG):
#                 a[1][i].clear()
#                 a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
#                 a[1][i].set_xticks(())
#                 a[1][i].set_yticks(())
#             plt.draw()
#             plt.pause(0.05)
#
# plt.ioff()
# plt.show()
