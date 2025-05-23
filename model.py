#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, num_cols=7, num_rows=6):
        super(ConvBlock, self).__init__()
        self.action_size = num_cols
        self.board_size = (num_rows, num_cols)
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s.view(-1, 3, self.board_size[0], self.board_size[1])
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
class OutBlock(nn.Module):
    def __init__(self, num_cols=7, num_rows=6):
        super(OutBlock, self).__init__()
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.conv = nn.Conv2d(128, 3, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3*num_rows*num_cols, 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.conv1 = nn.Conv2d(128, 32, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(self.num_rows*self.num_cols*32, self.num_cols)
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 3*self.num_rows*self.num_cols)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, self.num_rows*self.num_cols*32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v
    
class ConnectNet(nn.Module):
    def __init__(self, num_cols=7, num_rows=6, num_blocks=5):
        super(ConnectNet, self).__init__()
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.num_blocks = num_blocks
        self.conv = ConvBlock(num_cols=num_cols, num_rows=num_rows)
        for block in range(num_blocks):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock(num_cols=num_cols, num_rows=num_rows)
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(self.num_blocks):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s

class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, value_pred, value_target, policy_pred, policy_target):
        value_error = (value_pred - value_target) ** 2
        policy_error = torch.sum((-policy_target * (policy_pred + 1e-8).log()), 1)
        
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error