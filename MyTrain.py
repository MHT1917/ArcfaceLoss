import torch.nn as nn
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from MyNet import MainNet,Arcface
import torch.nn.functional as F
# from MyData import Mydata
# from MyNet_face import MainNet,CenterLoss

save_path = "models/net.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = torchvision.datasets.MNIST(root="MNIST",download=True,train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
train_loader = data.DataLoader(dataset=train_data,shuffle=True,batch_size=128)

# train_data = Mydata("datasets")
# train_loader = data.DataLoader(dataset=train_data,shuffle=True,batch_size=128)
def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    # plt.legend(['dlrb', 'lss', 'ym', 'sjl', 'tly', 'zjl', 'CR', 'wjk', 'hg', 'lx'], loc = 'upper right')
    # plt.xlim(xmin=-100,xmax=100)
    # plt.ylim(ymin=-100,ymax=100)
    plt.title("epoch=%d" % epoch)
    plt.savefig('./images/epoch=%d.jpg' % epoch)
    # plt.draw()
    # plt.pause(0.001)
# def getloss(outputs,features,labels):
#     center_loss_layer = CenterLoss(10,2).to(device)
#     CrossEntropyloss = nn.CrossEntropyLoss()
#     loss_cls = CrossEntropyloss(outputs,labels)
#     loss_center = center_loss_layer(features,labels)
#     loss = loss_cls+loss_center
#     return loss

if __name__ == '__main__':
    net =MainNet().to(device)
    if os.path.exists(save_path):
        net = torch.load(save_path)
    nllloss = nn.NLLLoss(reduction="sum").to(device)
    arcface = Arcface(2,10).to(device)

    # optimzer = torch.optim.SGD(net.parameters(),lr=0.001, weight_decay=0.0005)
    optimizer = torch.optim.Adam(net.parameters())

    # optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
    # sheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.8)
    optmizerarc = torch.optim.Adam(arcface.parameters())

    epoch = 0
    while True:
        # sheduler.step()
        print("epochs:{}".format(epoch))
        feat_loader = []
        label_loader = []
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            # target = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1).to(device)
            target = y.to(device)
            out_put,feat = net(x)

            # out_put = torch.argmax(out_put, dim=1).view(-1,1)

            arcloss = nllloss(torch.log(arcface(feat)),target)
            nll_loss = nllloss(out_put,target)

            loss = arcloss+nll_loss
            optimizer.zero_grad()
            optmizerarc.zero_grad()
            loss.backward()
            optimizer.step()
            optmizerarc.step()

            feat_loader.append(feat)
            label_loader.append((y))


            if i % 10 == 0:
                print(loss.item())
        feat = torch.cat(feat_loader, 0)
        labels = torch.cat(label_loader, 0)
        if epoch % 10 == 0:
            visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
        epoch+=1
        torch.save(net, save_path)

