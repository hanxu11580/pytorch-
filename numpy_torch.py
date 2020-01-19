import numpy as np
from torch.autograd import Variable
import torch

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)
'''
    variable 类似图纸
    requires_grad 计算梯度
'''
# print(tensor,variable)
t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)
# print(t_out,v_out)

# v_out.backward() #误差反向传递，tensor不行variable可以
# print(variable.grad) #查看梯度

# print(variable.data) #tensor形式

# print(variable.data.numpy()) #tensor形式转换成numpy形式