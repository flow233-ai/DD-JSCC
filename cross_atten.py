import numpy as np
import torch
import torch.nn as nn
import math
import

class cross_attention(nn.Module):
    def __init__(self):
        super(cross_attention,self).__init__()
        self.SNR_Tokens = torch.randn(32,7)
        self.Wq = torch.randn(32,32)
        self.Wk = torch.randn(32,32)
        self.Wv = torch.randn(32,32)
        self.Wo = torch.randn(32,32)
        self.W1 = torch.randn(32,20)
        self.W2 = torch.randn(20,32)
        self.b1 = torch.randn(20)
        self.b2 = torch.randn(32)

    def forward(self,input,SNR1,SNR2):
        x = input[0:(int)(input.shape[0]/2),:,:,:]
        #print(x.shape)
        y = input[(int)(input.shape[0]/2):input.shape[0],:,:,:]
        tempx = x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3])
        tempx = tempx.swapaxes(1,2)
        tempy = y.reshape(y.shape[0],y.shape[1],y.shape[2]*y.shape[3])
        tempy = tempy.swapaxes(1,2)
        #print(tempx.shape)
        tokenx = self.SNR_Tokens[:,(int)(SNR1/3)].repeat([x.shape[0],1,1])
        tokeny = self.SNR_Tokens[:,(int)(SNR2/3)].repeat([y.shape[0],1,1])
        #print(tempy.shape)
        Fx = torch.cat([tokenx,tempx],dim = 1)
        Fy = torch.cat([tokeny,tempy],dim = 1)
        #print(F.shape)
        m = torch.nn.LayerNorm(normalized_shape = [50,32])
        Fx = m(Fx)
        Fy = m(Fy)
        Qx = torch.matmul(Fx,self.Wq)
        Kx = torch.matmul(Fx,self.Wk)
        Vx = torch.matmul(Fx,self.Wv)
        Qy = torch.matmul(Fy,self.Wq)
        Ky = torch.matmul(Fy,self.Wk)
        Vy = torch.matmul(Fy,self.Wv)
        Ax = torch.matmul(Qx,torch.transpose(Ky,1,2))
        Ay = torch.matmul(Qy,torch.transpose(Kx,1,2))
        h1Fx = Fx + torch.matmul(torch.matmul(torch.nn.functional.softmax(Ax/math.sqrt(32)),Vx),self.Wo)
        h1Fy = Fy + torch.matmul(torch.matmul(torch.nn.functional.softmax(Ay/math.sqrt(32)),Vy),self.Wo)
        relu = nn.ReLU()
        temp = self.b1.repeat([x.shape[0],50,1])
        temp2 = self.b2.repeat([y.shape[0],50,1])
        h2Fx = h1Fx + torch.matmul(relu(torch.matmul(m(h1Fx),self.W1) + temp),self.W2) + temp2
        h2Fy = h1Fy + torch.matmul(relu(torch.matmul(m(h1Fy),self.W1) + temp),self.W2) + temp2
        h2F = torch.concat([h2Fx,h2Fy])[:,0:49,:]
        h2F = h2F.swapaxes(1,2).reshape(input.shape)
        return h2F
