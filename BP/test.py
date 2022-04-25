import numpy as np
import torch
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # normalization    #转化为-1到1
    x = x.reshape((-1,)) # flatten  #拉成一行  维度转化
    x = torch.from_numpy(x)
    return x
train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True)   #训练集
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)  #测试集


train_data = DataLoader(train_set, batch_size=64, shuffle=True)  #训练集
test_data = DataLoader(test_set, batch_size=128, shuffle=False)#测试集

net = nn.Sequential(    #网络层  四层网络了
    nn.Linear(784, 400),
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)


#定义loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),1e-1)  #学习率0.1


#开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []
for e in range(5):
    train_loss = 0
    train_acc = 0
    net.train()
    for im, label in train_data:
        im = Variable(im)   #变量
        label = Variable(label)
        #前向传播
        out = net(im)
        loss = criterion(out,label)
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #记录误差
        train_loss += loss.item ()
        #计算分类准确率
        _,pred = out.max(1)
        num_correct = (pred == label).sum().item ()
        acc = num_correct / im.shape[0]   #64
        train_acc += acc
    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))  #训练完所有数据的平均准确率

    # testing
    eval_loss = 0
    eval_acc = 0
    net.eval()  # 将模型改为预测模式
    for im, label in test_data:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item ()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item ()  #train_loss+=loss.item()  需要改成
        acc = num_correct / im.shape[0]
        eval_acc += acc
        # print(eval_loss)
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc:{:.6f}'.format(e, train_loss / len(train_data), train_acc / len(train_data),eval_loss / len(test_data), eval_acc / len(test_data)))

plt.title('train loss')
plt.plot(np.arange(len(losses)), losses)
plt.show()
plt.plot(np.arange(len(acces)), acces)
plt.title('train acc')
plt.show()
plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.title('test loss')
plt.show()
plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.title('test acc')
plt.show()
