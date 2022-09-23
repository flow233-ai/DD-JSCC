
import torch
import torchvision
import torch.nn as nn
import math
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
#import matplotlib.pyplot as plt
from torch.utils import data
import numpy as np
from torchvision import transforms
from deepjscc import DeepJSCC
from torchvision.utils import save_image

# 数据集的预处理
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

data_path = r'D:\paper coding\DD-JSCC(distributed)\data'
# 获取数据集
train_data = CIFAR10(data_path,train=True,transform=transform,download=False)
test_data = CIFAR10(data_path,train=False,transform=transform,download=False)
#train_data = train_data[0] #只取图片，不需要label
#test_data = test_data[0]

#迭代器生成
train_loader = data.DataLoader(train_data,batch_size=128,shuffle=True)

#导入训练好的模型参数
model = torch.load("./model/DD-JSCC.pth")

#样本可视化
from torchvision import transforms

#定义损失
loss_func = nn.MSELoss()

for epoch in range(1):
    for i,(x,y) in enumerate(train_loader):
        if(i > 6000):
            break
        if (i % 2 == 0):
            batch_x = Variable(x)  # torch.Size([128, 1, 28, 28])
            continue
        batch_x2 = Variable(x)
        #batch_y = Variable(y) # torch.Size([128])
        # 获取最后输出
        snr1 = 10
        snr2 = 20
        temp1 = batch_x[:, 0, :, :].unsqueeze(1)
        temp2 = batch_x2[:, 0, :, :].unsqueeze(1)
        out = model.forward1(temp1,temp2,snr1,snr2) # torch.Size([128,10])
        #print(out.shape)
        #print(torch.concat([temp1, temp2],dim = 0).shape)
        loss1 = loss_func(out, torch.concat([temp1, temp2],dim = 0))
        if i % 50 == 1:
            print(loss1)
        temp1 = batch_x[:, 1, :, :].unsqueeze(1)
        temp2 = batch_x2[:, 1, :, :].unsqueeze(1)
        out2 = model.forward1(temp1,temp2,snr1,snr2) # torch.Size([128,10])
        temp1 = batch_x[:, 2, :, :].unsqueeze(1)
        temp2 = batch_x2[:, 2, :, :].unsqueeze(1)
        out3 = model.forward1(temp1,temp2,snr1,snr2) # torch.Size([128,10])
        if i % 50 == 1:
            out = torch.cat([out, out2, out3], dim=1)
            save_image(batch_x[0], './picture/' + str(i) + 'a1.jpg')
            save_image(out[0], './picture/' + str(i) + 'b1.jpg')
            save_image(batch_x2[0], './picture/' + str(i) + 'a2.jpg')
            save_image(out[128], './picture/' + str(i) + 'b2.jpg')
