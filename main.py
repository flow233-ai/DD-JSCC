'''
目前已完成进度：
1. 完成JSCE、cross attention以及JSCD部分
2. 完成af_module的调整，完成图像生成模块
3. 使用GPU进行计算加速（还在研究当中）
'''

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

#计算PSNR指标
def PSNR(loss):
    return 10 * math.log10(1/loss)

def EachImg(img):
    img=img/2+0.5   #将图像数据转换为0.0->1.0之间，才能正常对比度显示（以前-1.0->1.0色调对比度过大）
    #plt.imshow(np.transpose(img,(1,2,0)))
    #plt.show()

#加速计算
torch.manual_seed(3407)
#CPU\GPU转换

# 数据集的预处理
transform = transforms.Compose([transforms.ToTensor()])

data_path = r'D:\paper coding\DD-JSCC(distributed)\data'
# 获取数据集
train_data = CIFAR10(data_path,train=True,transform=transform,download=False)
test_data = CIFAR10(data_path,train=False,transform=transform,download=False)

#样本可视化
image = train_data[0][0]
#print(image)
#EachImg(image)

#迭代器生成
train_loader = data.DataLoader(train_data,batch_size=128,shuffle=True)
# train_loader = data.DataLoader(train_data,batch_size=128,shuffle=True,num_workers=8)
test_loader = data.DataLoader(test_data,batch_size=100,shuffle=True)
# test_loader = data.DataLoader(test_data,batch_size=100,shuffle=True,num_workers=8)
 
#定义损失和优化器

#print(model.ratio) #查看信噪权值ratio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = DeepJSCC()
#model.to(device)
loss_func = nn.MSELoss()
loss_func2 = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(),lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)

if __name__ == '__main__':
    #训练网络
    loss_count = []
    for epoch in range(80):
        for i,(x,y) in enumerate(train_loader):
            if(i % 2 == 0):
                batch_x = Variable(x) # torch.Size([128, 1, 28, 28])
                continue
            batch_x2 = Variable(x)
            #batch_y = Variable(y) # torch.Size([128])
            # 获取最后输出
            #print(batch_x[:,0,:,:].unsqueeze(1).shape)

            #随机生成两个信道的信噪比
            snr1 = torch.randint(0,21,(1,)).item()
            snr2 = torch.randint(0,21,(1,)).item()

            temp1 = batch_x[:,0,:,:].unsqueeze(1)
            temp2 = batch_x2[:,0,:,:].unsqueeze(1)
            out1 = model.forward1(temp1,temp2,snr1,snr2) # torch.Size([128,10])
            loss1 = loss_func(out1,torch.concat([temp1,temp2]))
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss1.backward() # 误差反向传播，计算参数更新值
            opt.step() # 将参数更新值施加到net的parmeters上
            # 使用优化器优化损失

            temp1 = batch_x[:,1,:,:].unsqueeze(1)
            temp2 = batch_x2[:,1,:,:].unsqueeze(1)
            out2 = model.forward1(temp1,temp2,snr1,snr2) # torch.Size([128,10])
            loss2 = loss_func(out2,torch.concat([temp1,temp2]))
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss2.backward() # 误差反向传播，计算参数更新值
            opt.step() # 将参数更新值施加到net的parmeters上
            # 使用优化器优化损失

            temp1 = batch_x[:,2,:,:].unsqueeze(1)
            temp2 = batch_x2[:,2,:,:].unsqueeze(1)
            out3 = model.forward1(temp1,temp2,snr1,snr2) # torch.Size([128,10])
            loss3 = loss_func(out3,torch.concat([temp1,temp2]))
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss3.backward() # 误差反向传播，计算参数更新值
            opt.step() # 将参数更新值施加到net的parmeters上
            #opt.zero_grad()  # 清空上一步残余更新参数值
            #loss3.backward() # 误差反向传播，计算参数更新值
            #opt.step() # 将参数更新值施加到net的parmeters上
            # 使用优化器优化损失

            #temp1 = torch.concat([out1,out2,out3],dim=1)
            #out4 = model.forward2(temp1)
            #loss4 = 0.2 * loss_func2(out4,batch_y)
            #loss = loss1 + loss2 + loss3 + loss4

            if i % 60 == 1:
                loss_count.append(loss1.detach().numpy())
                print('{}:\t'.format(i), loss1.item(),'  ',loss2.item(),'  ',loss3.item())
                if epoch == 70:
                    out = torch.cat([out1, out2, out3], dim=1)
                    save_image(batch_x[0], './picture/' + str(i) + 'a1.jpg')
                    save_image(out[0], './picture/' + str(i) + 'b1.jpg')
                    save_image(batch_x2[0], './picture/' + str(i) + 'a2.jpg')
                    save_image(out[128], './picture/' + str(i) + 'b2.jpg')


        scheduler.step()

    print('PSNR:',PSNR((loss1+loss2+loss3)/3),'dB')
    torch.save(model,r'D:\paper coding\DD-JSCC(distributed)\model\DD-JSCC.pth')
    #plt.figure('PyTorch_CNN_Loss')
    #plt.plot(loss_count,label='Loss')
    #plt.legend()
    #plt.show()