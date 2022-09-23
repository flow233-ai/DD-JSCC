import torch
import math
import torch.nn as nn
from gdn import GDN
from fl_module import FL2
from af_module import AF
from cross_atten import cross_attention

class JSCD(nn.Module):
    def __init__(self):
        super(JSCD,self).__init__()
        self.fl1 = FL2(3,64,32,2,1)
        self.af1 = AF(32,2)
        self.croatt = cross_attention()
        self.fl2 = FL2(2,32,16,2,0)
        self.af2 = AF(16,2)
        self.fl3 = FL2(2,16,3,2,0)
        self.af3 = AF(3,2)
        self.trans_conv4 = nn.Sequential(
            nn.ConvTranspose2d(3,1,5,1,0)
        )
        self.igdn4 = GDN(1,'cpu',inverse=True)
        self.decoder_last = nn.Sigmoid()

    def forward(self,x1,x2,SNR1,SNR2):
        x = torch.concat([x1,x2],dim=0)
        x = self.fl1.forward(x)
        x = self.af1.forward2(x,SNR1,SNR2)
        x = self.croatt.forward(x,SNR1,SNR2) #互注意力机制
        x = self.fl2.forward(x)
        x = self.af2.forward2(x,SNR1,SNR2)
        x = self.fl3.forward(x)
        x = self.af3.forward2(x,SNR1,SNR2)
        x = self.trans_conv4(x)
        x = self.igdn4.forward(x)
        x = self.decoder_last(x)
        return x
