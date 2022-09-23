import torch
import math
import torch.nn as nn
from JSCE import JSCE
from JSCD import JSCD
from channel import channel_forward,channel_forward2

class DeepJSCC(nn.Module):
    def __init__(self):
        super(DeepJSCC,self).__init__()
        self.encoder1 = JSCE()
        self.encoder2 = JSCE()
        self.decoder1 = JSCD()

    def forward1(self,x1,x2,SNR1,SNR2):
        x1 = self.encoder1.forward(x1,SNR1)
        x1 = channel_forward(x1,math.sqrt(1 / (math.pow(10,SNR1/10))))
        x2 = self.encoder2.forward(x2,SNR2)
        x2 = channel_forward(x2,math.sqrt(1 / (math.pow(10,SNR2/10))))
        x = self.decoder1.forward(x1,x2,SNR1,SNR2)
        return x