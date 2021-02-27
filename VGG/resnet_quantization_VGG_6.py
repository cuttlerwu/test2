#########################################################################
#   cuttler.wu
#   ZJU
#   resnet18_quantization
#########################################################################

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
#from resnet_for_quantization import resnet18
#from ResNet18 import ResNet18
from torch.quantization import QuantStub, DeQuantStub, QConfig
import torch.quantization
import torch.utils.data
import torchvision.datasets
import numpy as np
import matplotlib.pyplot as plt
import time
import math

quant_n = np.ones(17)*8 #量化位宽  image层 + 16层
# quant_n[14:17] = np.ones(3)*16

scale_enable = np.bool(1)

scale_input = 1 #输入数据集
scale_base_weight = np.ones(16) #VGG16 共16层
scale_base_bias = np.ones(16) #VGG16 共16层
scale_base = np.ones(16) #VGG16 共16层

scale_layer_conv_weight = np.ones(13)  #layer.conv.weight
scale_layer_bn_bias = np.ones(13)      #layer.bn.bias
scale_fc_weight = np.ones(3)
scale_fc_bias = np.ones(3)
'''
# features
layer_conv = [0,3,7,10,14,17,20,24,27,30,34,37,40]
layer_bn = [1,4,8,11,15,18,21,25,28,31,35,38,41]
# classifier
layer_fc = [0,3,6] 
'''
#VGG16 Image classfication for cifar-10 with PyTorch 
#########################################################
def my_print(s,x,layer=0,cnt=0):
    x=x.float()
    #print('--------------------------------------------')
    if layer!=0:
        print('[layer cnt] = ', layer, ' ', cnt)
    print('%s'%(s),'的统计特性: min=',x.min(),'; max=',x.max(),'; mean=',x.mean())

#检测溢出
def my_detect_overflow(s,x,layer=0,cnt=0):
    x=x.float()
    if x.max()>1 or x.min()<-1:
        # print('----------------------')
        if layer!=0:
            print('[layer cnt] = ', layer, ' ', cnt)
        print('%s'%(s),' overflow min=',x.min(),' max=',x.max())

#8bit截取
def cut_out(x,quant_n):
    t=pow(2,quant_n-1)
    x.copy_((x*t).round().char().float()/t)
    
    return x

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
        x /= 4
        # my_print('fff0', x)
        
        x = self.fff1 (x)
        x /= 2
        # my_print('fff1', x)
        x = self.maxp0(x)
        x = self.fff2 (x)
        x /= 2
        # my_print('fff2', x)
        
        x = self.fff3 (x)
        x /= 4
        # my_print('fff3', x)
        
        x = self.maxp1(x)
        x = self.fff4 (x)
        x /= 4
        # my_print('fff4', x)
        
        x = self.fff5 (x)
        x /= 8
        # my_print('fff5', x)
        
        x = self.fff6 (x)
        x /= 4
        # my_print('fff6', x)
        
        x = self.maxp2(x)
        x = self.fff7 (x)
        x /= 4
        # my_print('fff7', x)
        
        x = self.fff8 (x)
        x /= 8
        # my_print('fff8', x)
        
        x = self.fff9 (x)
        x /= 8
        # my_print('fff9', x)
        
        x = self.maxp3(x)
        x = self.fff10(x)
        x /= 8
        # my_print('fff10', x)
        
        x = self.fff11(x)
        x /= 8
        # my_print('fff11', x)
        
        x = self.fff12(x)
        x /= 16
        # my_print('fff12', x)
        
        x = self.maxp4(x)

        x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.classifier0(x)
        x /= 128
        # my_print('classifier0', x)
        x = self.classifier1(x)
        x /= 16
        # my_print('classifier1', x)
        x = self.classifier2(x)
        # my_print('classifier2', x)
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
    



# Official utils
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res      

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    # print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')

cpu = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform_calibration = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # padding后随机裁剪
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#calibration_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=False, transform=transform_calibration)
calibration_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform_calibration)

calibrationloader = torch.utils.data.DataLoader(calibration_dataset, batch_size=256, shuffle=True)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (1-0.4914, 1-0.4822, 1-0.4465))
    ])
#testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform_test)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

#print('the Size (MB) of original resnet18:', os.path.getsize("./checkpoint/resnet18_weights.pth")/1e6)
# print('the Size (MB) of original resnet18:', os.path.getsize("../model/net_200.pth")/1e6)

net = vgg16().to(device)

net.load_state_dict(torch.load('../model/net_210_VGG.pth'))

# necessary
net.eval()
# print('resnet18.layer1: Before fusion \n')#, resnet18_model.layer1)

#replace_forward(resnet18_model)

#top1, top5 = evaluate(resnet18_model, testloader, cpu=cpu)
#print('Evaluation accuracy before quantization and fusion: %2.2f' % top1.avg)

#for name in resnet18_model.state_dict():print(name)

#计算输入数据集的量化scale，并存入scale_input
def cal_image_scale(quant_n):    
    ##################################
    # 对称量化
    tmp=0
    for image, target in testloader:    
        tmp=max(tmp, abs(image.min()),abs(image.max()))    
    max_quant=float(pow(2,quant_n-1)-1)/pow(2,quant_n-1)
    scale=max_quant/tmp
    global scale_input
    scale_input=scale
    # print('append scale_input:', scale)
    return scale

'''
################################################################
def cal_param():
    ###########################################################
    # 统计模型参数的特性
    ###########################################################
    x=np.array([],dtype=float)
    for pp in resnet18_model.parameters():
        #print(parameters)
        x=np.append(x,pp.cpu().detach().numpy())
        #x.sort()
    print('x.min=',x.min())
    print('x.max=',x.max())
    print('x.mean=',x.mean())
    print('|x|.max()=',max(abs(x.min()),abs(x.max())))
'''

################################################################
# signed integer 8-32bit: char short int long
def cal_scale(tmp,quant_n):      #n:量化位宽
    ##################################
    # 对称量化
    
    max_quant=float(pow(2,quant_n-1)-1)/pow(2,quant_n-1)
    scale=max_quant/tmp
    return scale

def quant(x, scale, quant_n):
    #param.copy_((param*scale).char().float())    
    #x.copy_((x*scale).float())   #quant
    t=pow(2,quant_n-1)
    #x.copy_((x*scale*t).round().char().float()/t)
    x.copy_((x*scale*t).round().int().float()/t)
    
    return x

def dequant(x, scale = 1) -> torch.Tensor:    
    x = x.float()
    x.copy_(x/scale)
    return x
    
# 简化batchnorm层
def simplify_bn(s):       
    ##########################################
    # ba0： weight*(x-mean)/var+bias
    #       weight_ * x + bias_
    ##########################################
    s = 'fff'+s+'.1'
    # print(s)
    for dd in net.state_dict():
        if s in dd and 'weight' in dd: 
            weight = net.state_dict()[dd]
        if s in dd and 'mean' in dd: 
            mean = net.state_dict()[dd]
        if s in dd and 'var' in dd: 
            var = net.state_dict()[dd]
        if s in dd and 'bias' in dd: 
            bias = net.state_dict()[dd]  
    # print('weight=',weight)
    # print('mean=',mean)
    # print('var=',var)
    # print('bias=',bias)
    
    bias.copy_(bias-weight*mean/torch.sqrt(var))
    weight.copy_(weight/torch.sqrt(var))
    mean.copy_(torch.zeros_like(mean))
    var.copy_(torch.ones_like(var)) 
    
    # print('weight=',weight)
    # print('mean=',mean)
    # print('var=',var)
    # print('bias=',bias)    


# conv+bn融合weight:融合后用conv做卷积，用bn层做+bias    
def fuse_conv_bn(s):  
    s1 = 'fff'+s+'.0.weight'  #conv
    s2 = 'fff'+s+'.1.weight'  #bn  
    # print(s1)
    # print(s2)
    for dd in net.state_dict():
        if s1 in dd: 
            conv_w = net.state_dict()[dd]
        if s2 in dd: 
            bn_w = net.state_dict()[dd]
            
    #[a,b,c,d]=conv_w.size()
    conv_w1 = conv_w.clone().view(conv_w.size()[0], -1)
    bn_w1 = torch.diag(bn_w)
    conv_w.copy_(torch.mm(bn_w1, conv_w1).view(conv_w.size())) 
    bn_w.copy_(torch.ones_like(bn_w)) #写1

#找出最大绝对值
def cal_max_param(s):    
    global scale_input
    global scale_first
    global scale_layer_conv_weight
    global scale_layer_bn_weight
    global scale_layer_bn_bias
    for dd in net.state_dict():
        if s in dd: 
            w = net.state_dict()[dd]
    r = max(abs(w.min()),abs(w.max())) 
    # print('%s'%(s),'的最大绝对值：', r)
    return r       
            
    
# 计算各层基础scale    
def cal_scale_base():   
    global scale_input
    global scale_base_weight
    global scale_base_bias
    global scale_base
    global scale_layer_conv_weight
    global scale_layer_bn_bias
    global scale_fc_weight
    global scale_fc_bias
    
    #layer层   
    for ii in range(0,13):
        scale_base_weight[ii] = cal_scale(cal_max_param('fff'+str(ii)+'.0.weight'), quant_n[ii+1])
        scale_base_bias[ii]   = cal_scale(cal_max_param('fff'+str(ii)+'.1.bias'), quant_n[ii+1])
    
    #fc层
    for ii in range(0,3):
        scale_base_weight[13+ii]  = cal_scale(cal_max_param('classifier'+str(ii)+'.0.weight'), quant_n[ii+14])
        scale_base_bias[13+ii]    = cal_scale(cal_max_param('classifier'+str(ii)+'.0.bias'), quant_n[ii+14])
                
def change_scale():    
    global scale_input
    global scale_base_weight
    global scale_base_bias
    global scale_base
    global scale_layer_conv_weight
    global scale_layer_bn_bias
    global scale_fc_weight
    global scale_fc_bias    

    # scale_layer_conv_weight[0]/=scale_input
    
    ###########################################################
    # layer_conv + layer_bn: [0]
    ###########################################################
    scale_base[0] = min(scale_base_weight[0], scale_base_bias[0]/scale_input*4)
    scale_layer_conv_weight[0] = scale_base[0]  
    scale_layer_bn_bias[0] = scale_base[0] 
    
    # 因x量化产生的补偿
    scale_layer_bn_bias[0]*=scale_input
    
    ##防止量化数据溢出，对应后续参数需要作出补偿除法
    # conv+bn 结果除以4，则相当于下一级的输入x/4，可补偿下一级bias
    ###########################################################
    # layer_conv + layer_bn: [1-12]
    ###########################################################
    for ii in range(1,13):
        if ii==6 or ii==9 or ii==10 or ii==11 or ii==12:
            scale_base[ii] = min(scale_base_weight[ii], scale_base_bias[ii]/scale_layer_bn_bias[ii-1]*8)
        elif ii==1 or ii==4 or ii==5 or ii==7 or ii==8:
            scale_base[ii] = min(scale_base_weight[ii], scale_base_bias[ii]/scale_layer_bn_bias[ii-1]*4)        
        else:
            scale_base[ii] = min(scale_base_weight[ii], scale_base_bias[ii]/scale_layer_bn_bias[ii-1]*2)
        scale_layer_conv_weight[ii] = scale_base[ii]
        scale_layer_bn_bias[ii] = scale_base[ii]    
        
        # 因x量化产生的补偿  
        scale_layer_bn_bias[ii] *= scale_layer_bn_bias[ii-1]
        # if scale_layer_bn_bias[ii-1]<=1:
            # scale_layer_bn_bias[ii] *= scale_layer_bn_bias[ii-1]
        # else:
            # scale_layer_conv_weight[ii] /= scale_layer_bn_bias[ii-1]
        
        ##防止量化数据溢出，对应后续参数需要作出补偿除法 
        if ii==6 or ii==9 or ii==10 or ii==11 or ii==12:
            scale_layer_bn_bias[ii]/=8  
        elif ii==1 or ii==4 or ii==5 or ii==7 or ii==8:
            scale_layer_bn_bias[ii]/=4  
        else:
            scale_layer_bn_bias[ii]/=2  
    ###########################################################
    # fc层
    ###########################################################
    scale_base[13] = min(scale_base_weight[13], scale_base_bias[13]/scale_layer_bn_bias[12]*16)
    scale_fc_weight[0] = scale_base[13]
    scale_fc_bias[0] = scale_base[13]    
    # 因x量化产生的补偿    
    scale_fc_bias[0] *= scale_layer_bn_bias[12]
    scale_fc_bias[0] /= 16
    
    scale_base[14] = min(scale_base_weight[14], scale_base_bias[14]/scale_fc_bias[0]*128)
    scale_fc_weight[1] = scale_base[14]
    scale_fc_bias[1] = scale_base[14]    
    # 因x量化产生的补偿    
    scale_fc_bias[1] *= scale_fc_bias[0]
    scale_fc_bias[1] /= 128
    
    scale_base[15] = min(scale_base_weight[15], scale_base_bias[15]/scale_fc_bias[1]*16)
    scale_fc_weight[2] = scale_base[15]
    scale_fc_bias[2] = scale_base[15]    
    # 因x量化产生的补偿    
    scale_fc_bias[2] *= scale_fc_bias[1]    
    scale_fc_bias[2] /= 16
    
#封装写入量化后参数
def write_p(s, scale, quant_n):    
    for dd in net.state_dict():
        if s in dd: 
            w = net.state_dict()[dd]
            # print('##########################################################')
            # print('quant ', dd)            
            # print("scale=", scale)   
            # my_print(s, w) 
            quant(w,scale,quant_n)    
            # my_print(s, w) 

#写入量化后参数    
def write_quant_param():        
    global scale_input    
    global scale_layer_conv_weight
    global scale_layer_bn_bias
    global scale_fc_weight
    global scale_fc_bias 
    
    #####################################################
    #print(net.state_dict()) #打印模型结构+参数值
    #print(net.state_dict) #打印模型结构 不带参数值    
    #for dd in net.state_dict():
    #    print(dd) #打印模型层名
    #####################################################
    
    ##########################################
    # layer_conv + layer_bn
    ##########################################
    for ii in range(0, 13):        
        write_p( ('fff'+str(ii)+'.0.weight'), scale_layer_conv_weight[ii], quant_n[ii+1])   
        write_p( ('fff'+str(ii)+'.1.bias'), scale_layer_bn_bias[ii], 8)       
    
    
    ###########################################################
    # layer_fc
    ###########################################################
    for ii in range(0, 3):        
        write_p( ('classifier'+str(ii)+'.0.weight'), scale_fc_weight[ii], quant_n[ii+14])   
        write_p( ('classifier'+str(ii)+'.0.bias'), scale_fc_bias[ii], 8)      
    
def evaluate(model, data_loader, eval_batches=10000, cpu=False):
    if cpu==0:
        net=model.to(device)
    else:
        net=model            
    net.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    
    with torch.no_grad():
        for image, target in data_loader:
            
            if cpu==0:
                image1=image.to(device)
                target=target.to(device)
            else:
                image1=image
                       
            if scale_enable:
                quant(image1,scale_input, quant_n[0])##### 量化 image 

            output = net(image1)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end='')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt > eval_batches:
                return top1, top5
    return top1, top5

def my_fuse():
    #融合bn层内部计算
    for i in range(0,13): 
        simplify_bn(str(i))
    
    # for dd in net.state_dict():
        # w = net.state_dict()[dd]
        # my_print(dd, w)
        
    #融合conv和bn
    for i in range(0,13): 
        fuse_conv_bn(str(i))
    
    # for dd in net.state_dict():
        # w = net.state_dict()[dd]
        # my_print(dd, w)
    # print("-----------------------------------------------------------------------")
    
    
def search(loopn):
    global quant_n
    bck_quant_n = quant_n.copy()
    ti = time.time()
    with open("log_search_VGG.txt", "w") as f6:  
        for jj in range(0,loopn):            
            max_top1=0
            max_ii = 0
            f6.write('Loop ')
            f6.write(str(jj+1))
            print('Loop ',jj+1)
            f6.write('================================\n')
            for ii in range(len(quant_n)-1,-1,-1):                   
                net.load_state_dict(torch.load('../model/net_210_VGG.pth'))
                net.eval()
                my_fuse()
                quant_n = bck_quant_n.copy()
                quant_n[ii]-=1
                if scale_enable:                      
                    cal_image_scale(quant_n[0])
                    cal_scale_base()
                    change_scale()
                    write_quant_param()
                top1, top5 = evaluate(net, testloader, cpu=0)
                for xx in quant_n:  
                    f6.write(str(xx))
                    f6.write(" ")
                f6.write('\n')
                f6.write('Evaluation accuracy before quantization and fusion:')
                f6.write(str(top1.avg))
                f6.write('\n')
                if top1.avg>max_top1:
                    max_top1 = top1.avg
                    max_ii = ii
            print('loop max_ii = ', max_ii, ' max_top1 = ', max_top1)
            f6.write('\n')
            f6.write('loop max_ii = ')
            f6.write(str(max_ii))
            f6.write(' max_top1 = ')
            f6.write(str(max_top1))
            bck_quant_n[max_ii]-=1
            new_ti = time.time()
            print('time = ', (new_ti-ti)/60, 'min' )
            f6.write('\n')
            f6.write('time = ')
            f6.write(str((new_ti-ti)/60))
            f6.write('\n')
            if max_top1<=90:
                break
        f6.close()





testn=32
yy=[]
xx=[]
for ii in range(1,33):
    quant_n = np.ones(27)*ii    
    #8bit量化
    net.load_state_dict(torch.load('../model/net_200_VGG.pth'))
    net.eval()
    my_fuse()
    if scale_enable:  
        cal_image_scale(quant_n[0])
        cal_scale_base()
        change_scale()
        write_quant_param()
    top1, top5 = evaluate(net, testloader, cpu=0)
    print(quant_n)
    print('Evaluation accuracy after 8 bit quantization: %2.2f' % top1.avg)
    yy.append(np.array(top1.avg.cpu()))
    xx.append(ii)
    testn-=1
    
zz=[]    
for ii in range(1,33):
    zz.append(92.89)
plt.plot(xx, yy,color='r',marker='o',linestyle='dashed',label='Quantization') 
plt.plot(xx, zz,color='b',marker='x',linestyle='dashed',label='FP32') 
plt.axis([0,32,0,100])
plt.xlabel('Quantization bits wide')
plt.ylabel('Test accuracy')
plt.title('Test accuracy of VGG16')
plt.annotate('FP32: 92.89',xy=(0,92.89),
            xytext=(1,70),arrowprops=dict(facecolor='black',shrink=0.01),
            ) 
plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
plt.show()  



plt.plot(xx, yy,color='r',marker='o',linestyle='dashed',label='Quantization') 
plt.plot(xx, zz,color='b',marker='x',linestyle='dashed',label='FP32') 
plt.axis([6,10,90,94])
plt.xlabel('Quantization bits wide')
plt.ylabel('Test accuracy')
plt.title('Test accuracy of VGG16')
# plt.annotate('FP32: 92.89',xy=(0,92.89),
            # xytext=(1,70),arrowprops=dict(facecolor='black',shrink=0.01),
            # ) 
plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
plt.show()      





#量化重训练
testn=32
quant_yy=[]
for ii in range(1,33):
    quant_n = np.ones(27)*ii    
    #8bit量化
    net.load_state_dict(torch.load('../model/net_210_VGG.pth'))
    net.eval()
    my_fuse()
    if scale_enable:  
        cal_image_scale(quant_n[0])
        cal_scale_base()
        change_scale()
        write_quant_param()
    top1, top5 = evaluate(net, testloader, cpu=0)
    print(quant_n)
    print('Evaluation accuracy after 8 bit quantization: %2.2f' % top1.avg)
    quant_yy.append(np.array(top1.avg.cpu()))
   
  
plt.plot(xx, yy,color='r',marker='o',linestyle='dashed',label='Quantization') 
plt.plot(xx, quant_yy,color='g',marker='+',linestyle='dashed',label='Quantization') 
plt.plot(xx, zz,color='b',marker='x',linestyle='dashed',label='FP32')
plt.axis([0,32,0,100])
plt.xlabel('Quantization bits wide')
plt.ylabel('Test accuracy')
plt.title('Test accuracy of Resnet18')
plt.annotate('FP32: 94.52',xy=(3,94.52),
            xytext=(1,70),arrowprops=dict(facecolor='black',shrink=0.01),
            ) 
plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
plt.show()       


plt.plot(xx, yy,color='r',marker='o',linestyle='dashed',label='Quantization') 
plt.plot(xx, quant_yy,color='g',marker='+',linestyle='dashed',label='retrain before Quantization') 
plt.plot(xx, zz,color='b',marker='x',linestyle='dashed',label='FP32') 
plt.axis([4,10,91,94])
plt.xlabel('Quantization bits wide')
plt.ylabel('Test accuracy')
plt.title('Test accuracy of VGG16')
# plt.annotate('FP32: 92.89',xy=(0,92.89),
            # xytext=(1,70),arrowprops=dict(facecolor='black',shrink=0.01),
            # ) 
#plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
plt.legend(loc=4, borderaxespad=0.)
plt.show()    
    
#search(300)

