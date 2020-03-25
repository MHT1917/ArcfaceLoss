import torch
import torch.nn as nn
import torch.nn.functional as F
class Arcface(nn.Module):
    def __init__(self,feature_dim,cls_dim):
        super(Arcface, self).__init__()
        self.w = nn.Parameter(torch.randn(feature_dim,cls_dim))
    def forward(self,feature):
        _w = F.normalize(self.w,dim=0)
        _x = F.normalize(feature,dim=1)
        cosa = torch.matmul(_x,_w)/10
        a = torch.acos(cosa)
        top = torch.exp(10*cosa)
        bottom = torch.sum(top,dim=1).unsqueeze(1)
        _top = torch.exp(10*torch.cos(a+0.1))
        out = _top/(bottom-top+_top)
        return out

class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2,1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 256, 3, 2,1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, 3, 1,1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 64, 3, 1,1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 16, 3, 2,1),
            nn.BatchNorm2d(16),
            nn.PReLU()
        )
        self.hidden_layer2 = nn.Linear(16 * 4 * 4, 2)
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(2,10,bias=False),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x.view(-1, 16 * 4 * 4))
        out = self.hidden_layer3(x)
        return out,x

if __name__ == '__main__':
    obj = Arcface(2,10)
    x = torch.Tensor(128,2)
    print(obj(x).shape)
