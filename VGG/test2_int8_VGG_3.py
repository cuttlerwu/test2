import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from quanlization_pytorch_master.ResNet18 import ResNet18
from quanlization_pytorch_master.resnet_for_quantization import resnet18
import os
import numpy as np
import math

quant_n = 9
scale = 1

# 定义是否使用GPU  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
args = parser.parse_args()

# 超参数设置
EPOCH = 205   #遍历数据集次数
pre_epoch = 200  # 定义已经遍历数据集的次数
BATCH_SIZE = 64      #批处理尺寸(batch_size)
LR = 0.0001        #学习率
  

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) #训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
class VGG(nn.Module):
    def __init__(self, fff, num_classes=10): #构造函数
        super(VGG, self).__init__()
        # 网络结构（仅包含卷积层和池化层，不包含分类器）
        self.fff0 = fff(3, 64)
        self.fff1 = fff(64, 64)
        self.maxp0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fff2 = fff(64, 128)
        self.fff3 = fff(128, 128)
        self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fff4 = fff(128, 256)
        self.fff5 = fff(256, 256)
        self.fff6 = fff(256, 256)
        self.maxp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fff7 = fff(256, 512)
        self.fff8 = fff(512, 512)
        self.fff9 = fff(512, 512)
        self.maxp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fff10 = fff(512, 512)
        self.fff11 = fff(512, 512)
        self.fff12 = fff(512, 512)
        self.maxp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier0 = nn.Sequential( #分类器结构
            #fc6
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            )
        self.classifier1 = nn.Sequential(
            #fc7
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            )
        self.classifier2 = nn.Sequential(
            #fc8
            nn.Linear(4096, num_classes))
        # 初始化权重
        self._initialize_weights()
 
    def forward(self, x):
        x = self.fff0 (x)
        x = self.fff1 (x)
        x = self.maxp0(x)
        x = self.fff2 (x)
        x = self.fff3 (x)
        x = self.maxp1(x)
        x = self.fff4 (x)
        x = self.fff5 (x)
        x = self.fff6 (x)
        x = self.maxp2(x)
        x = self.fff7 (x)
        x = self.fff8 (x)
        x = self.fff9 (x)
        x = self.maxp3(x)
        x = self.fff10(x)
        x = self.fff11(x)
        x = self.fff12(x)
        x = self.maxp4(x)

        x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.classifier0(x)
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x
 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

'''
# 生成网络每层的信息
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 设定卷积层的输出数量
            conv2d = nn.Conv2d(in_channels, v, 3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers) # 返回一个包含了网络结构的时序容器
'''

def fff(in_channels, out_channels):
    layers = []
    layers += [nn.Conv2d(in_channels, out_channels, 3, padding=1)]
    layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)
 
def vgg16(**kwargs):
    model = VGG(fff, **kwargs)
    #model.load_state_dict(torch.load('D:/my_project/myproject_python/quantization/model/vgg16-397923af.pth'))
    return model
    
# 模型定义-ResNet
net = vgg16().to(device)
net1 = vgg16().to(device)
#net = torchvision.models.resnet18(True).to(device)
#net = resnet18(pretrained=False).to(device)

#载入已保存的模型参数
net.load_state_dict(torch.load('./model/net_200_VGG.pth'))

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

sum_mark = 0  
for dd in net1.state_dict():
    if 'weight' in dd or 'bias' in dd:             
        w1 = net1.state_dict()[dd]
        w1.copy_(torch.ones_like(w1)) 
        sum_mark += w1.sum()
print('sum_mark = ', sum_mark) 
          
# n bit截取,伪量化
def cut_out():    
    sum_mark = 0 
    for dd in net.state_dict():
        if 'weight' in dd or 'bias' in dd: 
            w = net.state_dict()[dd]
            w1 = net1.state_dict()[dd]
            t=pow(2,quant_n-1)    
            w.copy_(w/scale)    
            w.copy_((w*t).float().round().float()/t)     
            w = w*w1
            w.copy_(w*scale)
            
            w1 = w1*w
            w1[w1!=0] = 1
            sum_mark += w1.sum()
    print('sum_mark = ', sum_mark) 
    
# 训练
if __name__ == "__main__":
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc = 85  #2 初始化best test accuracy
    print("Start Training, VGG16!")  # 定义遍历数据集的次数
    with open("acc_VGG.txt", "w") as f:
        with open("log_VGG.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    if i%100==0:
                        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()
                    
                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc_VGG.txt文件中
                    #print('Saving model......')
                    #torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc_VGG.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc_VGG.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
                        
                print('##########################################################')
                print('伪量化')                  
                cut_out()

                        
                # 每训练完一个epoch测试一下准确率
                print("Test again!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc_VGG.txt文件中
                    #print('Saving model......')
                    #torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("--- 伪量化 ---\n")
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc_VGG.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc_VGG.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print('Saving model......')
            torch.save(net.state_dict(), '%s/net_%03d_VGG.pth' % (args.outf, epoch + 1))        
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
                        

