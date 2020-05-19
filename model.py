import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from torch.autograd import Function
import numpy as np
import torch.nn.init as init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()
    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class PriorBox(object):
    def __init__(self):
        super(PriorBox, self).__init__()
        self.img_size=448
        self.clip=True
        self.variance=[0.1,0.2]
        self.features_map=[56,28,14,7,5,3]
        self.w_sizes = [0.13,0.22,0.15,0.30,0.22,0.39]
        self.h_sizes = [0.15,0.31,0.23,0.37,0.22,0.48]
        self.index=[[0,1,2,3],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[1,2,3,4],[2,3,4,5]]
        self.aspect_ratios=[4,6,6,6,4,4]
        self.step=[8,16,32,64,90,150]
    def forward(self):
       mean=[]
       for k,f in enumerate(self.features_map):
           x, y = np.meshgrid(np.arange(f), np.arange(f))
           x = x.reshape(-1)
           y = y.reshape(-1)
           for i,j in zip(y,x):
               f_k=self.img_size/self.step[k]
#              //计算网格中心
               cx=(j+0.5)/f_k
               cy=(i+0.5)/f_k
               index=self.index[k]
               for ic in index:
                   mean+=[cx,cy,self.w_sizes[ic],self.h_sizes[ic]]
       output = torch.Tensor(mean).view(-1, 4)
       if self.clip:
           output.clamp_(max=1,min=0)
       return output
class SSD(nn.Module):
    def __init__(self, phase, base, extras, head, num_classes,nms_thresh=0.5):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.nms_thresh=nms_thresh
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.priorbox=PriorBox()
        with torch.no_grad():
            self.priors=Variable(self.priorbox.forward().to(device))
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase=='test':
            self.softmax=nn.Softmax(dim=-1)
            self.detect=Detect(num_classes=self.num_classes,top_k=200,conf_thresh=0.1,nms_thresh=self.nms_thresh)
    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()
        # 获得conv4_3的内容
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # 获得fc7的内容
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # 获得后面的内容
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)



        # 添加回归层和分类层
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # 进行resize
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            # loc会resize到batch_size,num_anchors,4
            # conf会resize到batch_size,num_anchors,
             output=self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)), self.priors
            )

        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output
def add_vgg(i):
    base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]
    layers = []
    in_channels = i
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers
def add_extras(i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i

    # Block 6
    # 19,19,1024 -> 10,10,512
    layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

    # Block 7
    # 10,10,512 -> 5,5,256
    layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

    # Block 8
    # 5,5,256 -> 3,3,256
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
    # Block 9
    # 3,3,256 -> 1,1,256
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
    return layers
def get_ssd(phase,num_classes):
    vgg, extra_layers = add_vgg(3), add_extras(1024)
    mbox = [4, 6, 6, 6, 4, 4]
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 mbox[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  mbox[k] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                  * num_classes, kernel_size=3, padding=1)]
    SSD_MODEL = SSD(phase, vgg, extra_layers, (loc_layers, conf_layers), num_classes)
    return  SSD_MODEL
class Detect(Function):
    def __init__(self,num_classes,top_k,nms_thresh,conf_thresh):
        self.num_classes=num_classes
        self.top_k=top_k
        self.nms_thresh=nms_thresh
        self.conf_thresh=conf_thresh
        self.variances = [0.1, 0.2]
    def forward(self,loc_data,conf_data,prior_data):
        num=loc_data.size(0)  #其loc_data的batch_size
        num_priors=prior_data.size(0)
        output=torch.zeros(num,self.num_classes,self.top_k,5)
        conf_preds=conf_data.view(num,num_priors,self.num_classes).transpose(2, 1)
        #对每一张图片进行处理
        for i in range(num):
         #对先验框解码获得预测框
         decoded_boxes=decode(loc_data[i],prior_data,self.variances)
         conf_scores=conf_preds[i].clone()
         for cl in range(1,self.num_classes):
          #对每一类进行非极大抑制

          c_mask=conf_scores[cl].gt(0.01)
          scores=conf_scores[cl][c_mask]
          print(conf_scores[cl])
          if scores.size(0) == 0:
              continue
          l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
          boxes = decoded_boxes[l_mask].view(-1, 4)
          # 进行非极大抑制
          ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
          output[i, cl, :count] =torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)

          flt = output.contiguous().view(num, -1, 5)
          _, idx = flt[:, :, 0].sort(1, descending=True)
          _, rank = idx.sort(1)
          flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
          return output
def decode(loc,priors,variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes
def nms(boxes,scores,overlap=0.5,top_k=200):
    keep=scores.new(scores.size(0)).zero_().long()
    if boxes.numel()==0:
        return keep
    x1=boxes[:,0]
    y1=boxes[:,1]
    x2=boxes[:,2]
    y2=boxes[:,3]
    area=torch.mul(x2-x1,y2-y1)
    v,idx=scores.sort(0)
    idx = idx[-top_k:]
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union
        idx = idx[IoU.le(overlap)]
    print(keep)
    return keep, count
