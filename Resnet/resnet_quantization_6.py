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
from resnet_for_quantization import resnet18
#from ResNet18 import ResNet18
from torch.quantization import QuantStub, DeQuantStub, QConfig
import torch.quantization
import torch.utils.data
import torchvision.datasets
import numpy as np
import matplotlib.pyplot as plt
import time

quant_n = np.ones(27)*8 #量化位宽  image + first + 3*{layer1-4 (每个layer两层)} + fc  ；共11层
#quant_n = np.ones(27)*7 #量化位宽  image + first + 3*{layer1-4 (每个layer两层)} + fc  ；共11层

quant_n_layer = np.ones(8)*8 #layer1-4 (每个layer两层)

scale_enable = np.bool(1)

scale_input = 1 #输入数据集
scale_base = np.ones(10) #first + layer1-4 (每个layer两层) + fc  ；共10层

scale_first = np.ones(3) #co0.weight, bo0.weight, bo0.bias
scale_layer_conv_weight = np.ones(24)  #layer.conv.weight
#scale_layer_bn_weight = np.ones(24)
scale_layer_bn_bias = np.ones(24)      #layer.bn.bias

scale_fc_weight = np.array([1])
scale_fc_bias = np.array([1])

# 计算各层基础scale所对应的quant_n
def cal_quant_n_layer():
    global quant_n_layer
    for ii in range(0,8):
        quant_n_layer[ii] = min(quant_n[ii*3 + 2], quant_n[ii*3 + 3], quant_n[ii*3 + 4])

#ResNet-18 Image classfication for cifar-10 with PyTorch 
#########################################################
def my_print(s,x):
    x=x.float()
    #print('--------------------------------------------')
    print('%s'%(s),'的统计特性: min=',x.min(),'; max=',x.max(),'; mean=',x.mean())

#检测溢出
def my_detect_overflow(s,x):
    x=x.float()
    if x.max()>1 or x.min()<-1:
        # print('----------------------')
        print('%s'%(s),' overflow min=',x.min(),' max=',x.max())

#8bit截取
def cut_out(x,quant_n):
    t=pow(2,quant_n-1)
    x.copy_((x*t).round().char().float()/t)
    
    return x
    
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, layer=1, cnt=0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)        
        
        self.stride = stride
        self.layer = layer
        self.cnt = cnt
        
        self.conv3 = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(outchannel)
            

    def forward(self, x):
        out = x        
        
        # print('########################################')
        # print("layer = ", self.layer)
        # print('cnt = ', self.cnt)
        layer = self.layer
        cnt = self.cnt        
        
        out = self.conv1(out)  
        if scale_enable:
            out/=2         
        cut_out(out,8)#quant_n_layer[(layer-1)*2+cnt])    
        # my_detect_overflow('conv1:', out)
        
        out = self.bn1(out)                
        out = self.relu(out)
        out = self.conv2(out)
        if scale_enable:    
            if (layer==3 or layer==4) and cnt==0:
                out/=2   
          
        # my_detect_overflow('conv2:', out)
        
        out = self.bn2(out)  
    
        if self.stride:     
            shortcut = self.conv3(x) 
            if scale_enable:
                if (layer==3 or layer==4) and cnt==0:
                    shortcut/=4  
                else:
                    shortcut/=2  
            # my_detect_overflow('conv3:', shortcut)
            
            shortcut = self.bn3(shortcut)        #注意：输出量化编码时这路分支用shortcut命名 
            out += shortcut
            cut_out(out,8)#quant_n_layer[(layer-1)*2+cnt])    
        out = F.relu(out)
        # if scale_enable:
            # if layer==4 and cnt==1:
                # out*=16   
                
        # if scale_enable:
            # if layer==4 and cnt==1:
                # dequant(out, scale_layer_bn_bias[11])
        
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        '''
        #第一层用单独的缩写层名        
        self.co0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)        
        self.ba0 = nn.BatchNorm2d(64)
        self.re0 = nn.ReLU()
        
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1, layer=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2, layer=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2, layer=3)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2, layer=4)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride, layer):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        
        cnt=0
        for stride in strides:           
            layers.append(block(self.inchannel, channels, stride, layer, cnt))
            cnt+=1
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        
        out = x
        #cut_out(out,quant_n[0])
        
        # my_print('input :', out)
        out = self.co0(out)
        ##防止量化数据溢出，对应后续参数需要作出补偿除法
        if scale_enable:
            out/=2          
        cut_out(out,8)#quant_n[1])
        out = self.ba0(out)
        # my_print('ba0  :', out)
        out = self.re0(out)  
        #my_detect_overflow('first layer re0 :', out)        

        out = self.layer1(out)        
        # my_print('layer1:', out)
        out = self.layer2(out)
        # my_print('layer2:', out)
        out = self.layer3(out)
        # my_print('layer3:', out)
        out = self.layer4(out)
        # my_print('layer4:', out)        
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # my_print('fc:', out)      

        return out


def ResNet18():
    return ResNet(ResidualBlock)


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
#resnet18_model = resnet18(pretrained=True, pretrained_model_path='../model/net_200.pth') #原始载入模型语句
#resnet18_model = resnet18(pretrained=True)
resnet18_model = ResNet18()
#resnet18_model = torchvision.models.resnet18()
resnet18_model.load_state_dict(torch.load('../model/net_210.pth'))

# necessary
resnet18_model.eval()
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
    x.copy_((x*scale*t).round().long().float()/t)
    
    return x

def dequant(x, scale = 1) -> torch.Tensor:    
    x = x.float()
    x.copy_(x/scale)
    return x
    
# 简化batchnorm层
def simplify_bn(s):
    net = resnet18_model 
    
    ##########################################
    # ba0： weight*(x-mean)/var+bias
    #       weight_ * x + bias_
    ##########################################
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
    

# 简化 layer1-4的batchnorm层
def simplify_bn2(s1,s2):
    net = resnet18_model 
    
    ##########################################
    # ba0： weight*(x-mean)/var+bias
    #       weight_ * x + bias_
    ##########################################
    for dd in net.state_dict():
        if s1 in dd and s2 in dd and 'weight' in dd: 
            weight = net.state_dict()[dd]
        if s1 in dd and s2 in dd and 'mean' in dd: 
            mean = net.state_dict()[dd]
        if s1 in dd and s2 in dd and 'var' in dd: 
            var = net.state_dict()[dd]
        if s1 in dd and s2 in dd and 'bias' in dd: 
            bias = net.state_dict()[dd]  
    # print('weight=',weight)
    # print('mean=',mean)
    # print('var=',var)
    # print('bias=',bias)    
    
    bias.copy_(bias-weight*mean/torch.sqrt(var))
    weight.copy_(weight/torch.sqrt(var))
    mean.copy_(torch.zeros_like(mean))  #mean=0
    var.copy_(torch.ones_like(var))     #var=1
    # print('weight=',weight)
    # print('mean=',mean)
    # print('var=',var)
    # print('bias=',bias)

# conv+bn融合weight:融合后用conv做卷积，用bn层做+bias    
def fuse_conv_bn(s1,s2):
    net = resnet18_model 
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
    net = resnet18_model 
    
    global scale_input
    global scale_base
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
            
def cal_scale_base_conv(s, quant_n):
    tmp1 = cal_scale(cal_max_param(s+'.conv1.weight'),quant_n)    
    tmp2 = cal_scale(cal_max_param(s+'.conv2.weight'),quant_n)    
    tmp3 = cal_scale(cal_max_param(s+'.conv3.weight'),quant_n)    
    tmp4 = cal_scale(cal_max_param(s+'.bn1.bias'),quant_n)    
    tmp5 = cal_scale(cal_max_param(s+'.bn2.bias'),quant_n)    
    tmp6 = cal_scale(cal_max_param(s+'.bn3.bias'),quant_n)   
    # return min(tmp1, tmp2, tmp3.sqrt(), tmp4, tmp5.sqrt(), tmp6.sqrt())
    return min(tmp1, tmp2, tmp3.sqrt())
    
# 计算各层基础scale    
def cal_scale_base():
    net = resnet18_model 
    
    global scale_input
    global scale_base
    global scale_first
    global scale_layer_conv_weight
    global scale_layer_bn_weight
    global scale_layer_bn_bias
    
    #first层
    tmp1 = cal_scale(cal_max_param('co0.weight'),quant_n[1])        
    tmp2 = cal_scale(cal_max_param('ba0.bias'),quant_n[1])    
    scale_base[0] = tmp1#min(tmp1, tmp2)
    
    #layer层    
    scale_base[1] = cal_scale_base_conv('layer1.0', quant_n_layer[0])  
    scale_base[2] = cal_scale_base_conv('layer1.1', quant_n_layer[1])  
    scale_base[3] = cal_scale_base_conv('layer2.0', quant_n_layer[2])  
    scale_base[4] = cal_scale_base_conv('layer2.1', quant_n_layer[3])  
    scale_base[5] = cal_scale_base_conv('layer3.0', quant_n_layer[4])  
    scale_base[6] = cal_scale_base_conv('layer3.1', quant_n_layer[5])  
    scale_base[7] = cal_scale_base_conv('layer4.0', quant_n_layer[6])  
    scale_base[8] = cal_scale_base_conv('layer4.1', quant_n_layer[7])    
    
    scale_base[0] *= scale_input #防止量化补偿后溢出
    # scale_base *= np.array(scale_input)
    
    #fc层
    # tmp=max(
        # cal_max_param('fc.weight'),        
        # cal_max_param('fc.bias')
        # )
    tmp = cal_max_param('fc.weight')
    scale_base[9] = cal_scale(tmp,quant_n[26])    
    
def change_scale():
    net = resnet18_model 
    
    global scale_input
    global scale_base
    global scale_first
    global scale_layer_conv_weight
    global scale_layer_bn_weight
    global scale_layer_bn_bias
    global scale_fc_weight
    global scale_fc_bias
    
    ###########################################################
    #first层
    ###########################################################    
    #基础量化值
    scale_first[0] = scale_base[0]
    #scale_first[1] = scale_base[0] 融合后不需要
    scale_first[2] = scale_base[0]  #bo0.bias
    
    # 因input量化产生的补偿   #融合后不需要补偿bn特性的系数scale_base[0] 
    scale_first[2] *= scale_input    
    # scale_first[0] /= scale_input #如果补偿scale_first[2]则做乘法，但是后续参数积累之下会溢出
        
    ##防止量化数据溢出，对应后续参数需要作出补偿除法
    scale_first[2] /= 2 
    ###########################################################
    #layer1.0层
    ###########################################################
    scale_layer_conv_weight[0] = scale_base[1]  
    scale_layer_conv_weight[1] = scale_base[1]    
    scale_layer_conv_weight[2] = scale_base[1]*scale_base[1]    
    
    scale_layer_bn_bias[0] = scale_base[1] 
    scale_layer_bn_bias[1] = scale_base[1]*scale_base[1]       
    scale_layer_bn_bias[2] = scale_base[1]*scale_base[1]    
    
    # 因x量化产生的补偿
    # scale_layer_conv_weight[0] /= scale_first[2]
    # scale_layer_conv_weight[2] /= scale_first[2]
    scale_layer_bn_bias[0]*=scale_first[2]
    scale_layer_bn_bias[1]*=scale_first[2]
    scale_layer_bn_bias[2]*=scale_first[2]
    
    ##防止量化数据溢出，对应后续参数需要作出补偿除法
    scale_layer_bn_bias[0]/=2
    scale_layer_bn_bias[1]/=2
    scale_layer_bn_bias[2]/=2
    
    #注明：scale_layer_bn_bias[1]==scale_layer_bn_bias[2] ，可以合并相加
    
    ###########################################################
    # layer1.1-layer4.1 用循环实现
    ###########################################################
    for ii in range(1,8):
        scale_layer_conv_weight[3*ii] = scale_base[ii+1]    
        scale_layer_conv_weight[3*ii+1] = scale_base[ii+1]    
        scale_layer_conv_weight[3*ii+2] = scale_base[ii+1]*scale_base[ii+1] 
        
        scale_layer_bn_bias[3*ii] = scale_base[ii+1]    
        scale_layer_bn_bias[3*ii+1] = scale_base[ii+1]*scale_base[ii+1] 
        scale_layer_bn_bias[3*ii+2] = scale_base[ii+1]*scale_base[ii+1] 
        
        # 因x量化产生的补偿
        # scale_layer_conv_weight[3*ii] /= scale_layer_bn_bias[3*ii-1]
        # scale_layer_conv_weight[3*ii+2] /= scale_layer_bn_bias[3*ii-1]
        scale_layer_bn_bias[3*ii]   *= scale_layer_bn_bias[3*ii-1]
        scale_layer_bn_bias[3*ii+1] *= scale_layer_bn_bias[3*ii-1]
        scale_layer_bn_bias[3*ii+2] *= scale_layer_bn_bias[3*ii-1]      
        
        ##防止量化数据溢出，对应后续参数需要作出补偿除法        
        #if ii%2==1:
        if ii==4 or ii==6:
            scale_layer_bn_bias[3*ii]/=2
            scale_layer_bn_bias[3*ii+1]/=4
            scale_layer_bn_bias[3*ii+2]/=4
        else:
            scale_layer_bn_bias[3*ii]/=2
            scale_layer_bn_bias[3*ii+1]/=2
            scale_layer_bn_bias[3*ii+2]/=2
        
        #注明：scale_layer_bn_bias[3*ii+1]==scale_layer_bn_bias[3*ii+2] ，可以合并相加
        
    ###########################################################
    # fc层
    ###########################################################
    scale_fc_weight = scale_base[9]
    scale_fc_bias = scale_base[9]
    # 因x量化产生的补偿
    # scale_fc_weight /= scale_layer_bn_bias[23]
    scale_fc_bias *= scale_layer_bn_bias[23]
 
    
#封装写入量化后参数
def write_p(s, scale, quant_n):
    net = resnet18_model 
    for dd in net.state_dict():
        if s in dd: 
            w = net.state_dict()[dd]
            # print('##########################################################')
            # print('quant ', dd)            
            # print("scale=", scale)             
            quant(w,scale,quant_n)    
            # my_print(s, w) 

#写入量化后参数    
def write_quant_param():
    net = resnet18_model 
    
    global scale_input
    global scale_base
    global scale_first
    global scale_layer_conv_weight
    global scale_layer_bn_weight
    global scale_layer_bn_bias
    
    #####################################################
    #print(net.state_dict()) #打印模型结构+参数值
    #print(net.state_dict) #打印模型结构 不带参数值    
    #for dd in net.state_dict():
    #    print(dd) #打印模型层名
    #####################################################
    
    ##########################################
    # first: co0 + ba0
    ##########################################
    write_p( 'co0.weight', scale_first[0], quant_n[1])   
    write_p( 'ba0.bias', scale_first[2], 8)  
    
    ###########################################################
    #layer1.0层
    ###########################################################
    write_p( 'layer1.0.conv1.weight', scale_layer_conv_weight[0], quant_n[2]) 
    write_p( 'layer1.0.bn1.bias', scale_layer_bn_bias[0], 8)  
    write_p( 'layer1.0.conv2.weight', scale_layer_conv_weight[1], quant_n[3]) 
    write_p( 'layer1.0.bn2.bias', scale_layer_bn_bias[1], 8)  
    write_p( 'layer1.0.conv3.weight', scale_layer_conv_weight[2], quant_n[4]) 
    write_p( 'layer1.0.bn3.bias', scale_layer_bn_bias[2], 8)    
    
    ###########################################################
    #layer1.1层
    ###########################################################
    write_p( 'layer1.1.conv1.weight', scale_layer_conv_weight[3], quant_n[5]) 
    write_p( 'layer1.1.bn1.bias', scale_layer_bn_bias[3], 8)  
    write_p( 'layer1.1.conv2.weight', scale_layer_conv_weight[4], quant_n[6]) 
    write_p( 'layer1.1.bn2.bias', scale_layer_bn_bias[4], 8) 
    write_p( 'layer1.1.conv3.weight', scale_layer_conv_weight[5], quant_n[7]) 
    write_p( 'layer1.1.bn3.bias', scale_layer_bn_bias[5], 8)    

    
    ###########################################################
    #layer2.0层
    ###########################################################
    write_p( 'layer2.0.conv1.weight', scale_layer_conv_weight[6], quant_n[8]) 
    write_p( 'layer2.0.bn1.bias', scale_layer_bn_bias[6], 8) 
    write_p( 'layer2.0.conv2.weight', scale_layer_conv_weight[7], quant_n[9]) 
    write_p( 'layer2.0.bn2.bias', scale_layer_bn_bias[7], 8)    
    write_p( 'layer2.0.conv3.weight', scale_layer_conv_weight[8], quant_n[10]) 
    write_p( 'layer2.0.bn3.bias', scale_layer_bn_bias[8], 8)    
    
    ###########################################################
    #layer2.1层
    ###########################################################
    write_p( 'layer2.1.conv1.weight', scale_layer_conv_weight[9], quant_n[11]) 
    write_p( 'layer2.1.bn1.bias', scale_layer_bn_bias[9], 8)  
    write_p( 'layer2.1.conv2.weight', scale_layer_conv_weight[10], quant_n[12]) 
    write_p( 'layer2.1.bn2.bias', scale_layer_bn_bias[10], 8)   
    write_p( 'layer2.1.conv3.weight', scale_layer_conv_weight[11], quant_n[13]) 
    write_p( 'layer2.1.bn3.bias', scale_layer_bn_bias[11], 8)    
    
    ###########################################################
    #layer3.0层
    ###########################################################
    write_p( 'layer3.0.conv1.weight', scale_layer_conv_weight[12], quant_n[14]) 
    write_p( 'layer3.0.bn1.bias', scale_layer_bn_bias[12], 8) 
    write_p( 'layer3.0.conv2.weight', scale_layer_conv_weight[13], quant_n[15]) 
    write_p( 'layer3.0.bn2.bias', scale_layer_bn_bias[13], 8)    
    write_p( 'layer3.0.conv3.weight', scale_layer_conv_weight[14], quant_n[16]) 
    write_p( 'layer3.0.bn3.bias', scale_layer_bn_bias[14], 8)    
    
    ###########################################################
    #layer3.1层
    ###########################################################
    write_p( 'layer3.1.conv1.weight', scale_layer_conv_weight[15], quant_n[17]) 
    write_p( 'layer3.1.bn1.bias', scale_layer_bn_bias[15], 8)  
    write_p( 'layer3.1.conv2.weight', scale_layer_conv_weight[16], quant_n[18]) 
    write_p( 'layer3.1.bn2.bias', scale_layer_bn_bias[16], 8)   
    write_p( 'layer3.1.conv3.weight', scale_layer_conv_weight[17], quant_n[19]) 
    write_p( 'layer3.1.bn3.bias', scale_layer_bn_bias[17], 8)     
    
    ###########################################################
    #layer4.0层
    ###########################################################
    write_p( 'layer4.0.conv1.weight', scale_layer_conv_weight[18], quant_n[20]) 
    write_p( 'layer4.0.bn1.bias', scale_layer_bn_bias[18], 8) 
    write_p( 'layer4.0.conv2.weight', scale_layer_conv_weight[19], quant_n[21]) 
    write_p( 'layer4.0.bn2.bias', scale_layer_bn_bias[19], 8)    
    write_p( 'layer4.0.conv3.weight', scale_layer_conv_weight[20], quant_n[22]) 
    write_p( 'layer4.0.bn3.bias', scale_layer_bn_bias[20], 8)    
    
    ###########################################################
    #layer4.1层
    ###########################################################
    write_p( 'layer4.1.conv1.weight', scale_layer_conv_weight[21], quant_n[23]) 
    write_p( 'layer4.1.bn1.bias', scale_layer_bn_bias[21], 8)  
    write_p( 'layer4.1.conv2.weight', scale_layer_conv_weight[22], quant_n[24]) 
    write_p( 'layer4.1.bn2.bias', scale_layer_bn_bias[22], 8)   
    write_p( 'layer4.1.conv3.weight', scale_layer_conv_weight[23], quant_n[25]) 
    write_p( 'layer4.1.bn3.bias', scale_layer_bn_bias[23], 8)    
    
    ###########################################################
    #fc层
    ###########################################################
    write_p( 'fc.weight', scale_fc_weight, quant_n[26]) 
    write_p( 'fc.bias', scale_fc_bias, 8) 
    
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
    simplify_bn('ba')
    simplify_bn2('layer1.0','bn1')
    simplify_bn2('layer1.0','bn2')
    simplify_bn2('layer1.0','bn3')
    simplify_bn2('layer1.1','bn1')
    simplify_bn2('layer1.1','bn2')
    simplify_bn2('layer1.1','bn3')
    simplify_bn2('layer2.0','bn1')
    simplify_bn2('layer2.0','bn2')
    simplify_bn2('layer2.0','bn3')
    simplify_bn2('layer2.1','bn1')
    simplify_bn2('layer2.1','bn2')
    simplify_bn2('layer2.1','bn3')
    simplify_bn2('layer3.0','bn1')
    simplify_bn2('layer3.0','bn2')
    simplify_bn2('layer3.0','bn3')
    simplify_bn2('layer3.1','bn1')
    simplify_bn2('layer3.1','bn2')
    simplify_bn2('layer3.1','bn3')
    simplify_bn2('layer4.0','bn1')
    simplify_bn2('layer4.0','bn2')
    simplify_bn2('layer4.0','bn3')
    simplify_bn2('layer4.1','bn1')
    simplify_bn2('layer4.1','bn2')
    simplify_bn2('layer4.1','bn3')
    
    #融合conv和bn
    fuse_conv_bn('co0.weight','ba0.weight')
    fuse_conv_bn('layer1.0.conv1.weight','layer1.0.bn1.weight')
    fuse_conv_bn('layer1.0.conv2.weight','layer1.0.bn2.weight')
    fuse_conv_bn('layer1.0.conv3.weight','layer1.0.bn3.weight')
    fuse_conv_bn('layer1.1.conv1.weight','layer1.1.bn1.weight')
    fuse_conv_bn('layer1.1.conv2.weight','layer1.1.bn2.weight')
    fuse_conv_bn('layer1.1.conv3.weight','layer1.1.bn3.weight')
    fuse_conv_bn('layer2.0.conv1.weight','layer2.0.bn1.weight')
    fuse_conv_bn('layer2.0.conv2.weight','layer2.0.bn2.weight')
    fuse_conv_bn('layer2.0.conv3.weight','layer2.0.bn3.weight')
    fuse_conv_bn('layer2.1.conv1.weight','layer2.1.bn1.weight')
    fuse_conv_bn('layer2.1.conv2.weight','layer2.1.bn2.weight')
    fuse_conv_bn('layer2.1.conv3.weight','layer2.1.bn3.weight')
    fuse_conv_bn('layer3.0.conv1.weight','layer3.0.bn1.weight')
    fuse_conv_bn('layer3.0.conv2.weight','layer3.0.bn2.weight')
    fuse_conv_bn('layer3.0.conv3.weight','layer3.0.bn3.weight')
    fuse_conv_bn('layer3.1.conv1.weight','layer3.1.bn1.weight')
    fuse_conv_bn('layer3.1.conv2.weight','layer3.1.bn2.weight')
    fuse_conv_bn('layer3.1.conv3.weight','layer3.1.bn3.weight')
    fuse_conv_bn('layer4.0.conv1.weight','layer4.0.bn1.weight')
    fuse_conv_bn('layer4.0.conv2.weight','layer4.0.bn2.weight')
    fuse_conv_bn('layer4.0.conv3.weight','layer4.0.bn3.weight')
    fuse_conv_bn('layer4.1.conv1.weight','layer4.1.bn1.weight')
    fuse_conv_bn('layer4.1.conv2.weight','layer4.1.bn2.weight')
    fuse_conv_bn('layer4.1.conv3.weight','layer4.1.bn3.weight')


def search(loopn):
    global quant_n
    global resnet18_model
    bck_quant_n = quant_n.copy()
    ti = time.time()
    with open("log_search.txt", "w") as f6:  
        for jj in range(0,loopn):            
            max_top1=0
            max_ii = 0
            f6.write('Loop ')
            f6.write(str(jj+1))
            print('Loop ',jj+1)
            f6.write('================================\n')
            for ii in range(len(quant_n)-1,-1,-1):
                resnet18_model = ResNet18()        
                resnet18_model.load_state_dict(torch.load('../model/net_210.pth'))
                resnet18_model.eval()
                my_fuse()
                quant_n = bck_quant_n.copy()
                quant_n[ii]-=1
                if scale_enable:  
                    cal_quant_n_layer()
                    cal_image_scale(quant_n[0])
                    cal_scale_base()
                    change_scale()
                    write_quant_param()
                top1, top5 = evaluate(resnet18_model, testloader, cpu=0)
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
            if max_top1<=92:
                break
        f6.close()

testn=32
yy=[]
xx=[]
for ii in range(1,33):
    quant_n = np.ones(27)*ii    
    #8bit量化
    resnet18_model = ResNet18()        
    resnet18_model.load_state_dict(torch.load('../model/net_200.pth'))
    resnet18_model.eval()
    my_fuse()
    if scale_enable:    
        cal_quant_n_layer()
        cal_image_scale(quant_n[0])
        cal_scale_base()
        change_scale()
        write_quant_param()
    top1, top5 = evaluate(resnet18_model, testloader, cpu=0)
    print(quant_n)
    print('Evaluation accuracy after 8 bit quantization: %2.2f' % top1.avg)
    yy.append(np.array(top1.avg.cpu()))
    xx.append(ii)    
    
zz=[]    
for ii in range(1,33):
    zz.append(94.52)
plt.plot(xx, yy,color='r',marker='o',linestyle='dashed',label='Quantization') 
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
plt.plot(xx, zz,color='b',marker='x',linestyle='dashed',label='FP32') 
plt.axis([6,10,90,95])
plt.xlabel('Quantization bits wide')
plt.ylabel('Test accuracy')
plt.title('Test accuracy of Resnet18')
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
    resnet18_model = ResNet18()        
    resnet18_model.load_state_dict(torch.load('../model/net_210.pth'))
    resnet18_model.eval()
    my_fuse()
    if scale_enable:    
        cal_quant_n_layer()
        cal_image_scale(quant_n[0])
        cal_scale_base()
        change_scale()
        write_quant_param()
    top1, top5 = evaluate(resnet18_model, testloader, cpu=0)
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
plt.axis([6,10,92,95])
plt.xlabel('Quantization bits wide')
plt.ylabel('Test accuracy')
plt.title('Test accuracy of Resnet18')
# plt.annotate('FP32: 92.89',xy=(0,92.89),
            # xytext=(1,70),arrowprops=dict(facecolor='black',shrink=0.01),
            # ) 
#plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
plt.legend(loc=4, borderaxespad=0.)
plt.show()    


#search(300)

