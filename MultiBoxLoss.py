import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
np.set_printoptions(threshold=np.inf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def point_form(boxes):
    return torch.cat((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),1)
def intersect(box_a,box_b):
    A=box_a.size(0)
    B=box_b.size(0)
    max_xy=torch.min(box_a[:,2:].unsqueeze(1).expand(A, B, 2),box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy=torch.max(box_a[:,:2].unsqueeze(1).expand(A, B, 2),box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter=torch.clamp((max_xy-min_xy),min=0)
    return inter[:,:,0]*inter[:,:,1]
def jaccard(box_a,box_b):
     inter=intersect(box_a,box_b)
    # #计算先验框和真实框各自的面积
     area_a=((box_a[:,2]-box_a[:,0])*(box_a[:,3]-box_a[:,1])).unsqueeze(1).expand_as(inter)
     area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
     union=area_a+area_b-inter
     return inter/union
def match(threshold,truths,priors,labels,variances,loc_t,conf_t,idx):
  #计算所有的先验框
   overlaps=jaccard(point_form(truths),point_form(priors))
  #所有真实框和先验框的最好重合程度
   best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
   best_prior_idx.squeeze_(1)
   best_prior_overlap.squeeze_(1)
   best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
   best_truth_idx.squeeze_(0)
   best_truth_overlap.squeeze_(0)
   best_truth_overlap.index_fill_(0, best_prior_idx, 2)
   # 对best_truth_idx内容进行设置 保障每个先验框与真实框对应起来
   for j in range(best_prior_idx.size(0)):
       best_truth_idx[best_prior_idx[j]] =j
   matches=truths[best_truth_idx]
   conf=labels[best_truth_idx]
   order = torch.argsort(best_truth_overlap, dim=0, descending=True)
   conf[best_truth_overlap < threshold] = 0
   conf[order[:3]] = labels[best_truth_idx[order[:3]]]
   loc = encode(matches, priors, variances)
   loc_t[idx]=loc
   conf_t[idx]=conf
def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max
def encode(matched, priors,variances):
    g_cxcy=(matched[:,:2]-priors[:,:2])
    g_cxcy/=(priors[:,2:]+variances[0])
    g_wh=(matched[:, 2:]) / priors[:, 2:]
    g_wh=torch.log(g_wh)/variances[1]
    return torch.cat([g_cxcy,g_wh],1)
class MultiBoxloss(nn.Module):
    def __init__(self,num_class,overlap_thresh,neg_pos,use_gpu=True):
        super(MultiBoxloss, self).__init__()
        self.use_gpu=use_gpu
        self.num_classes=num_class
        self.threshold=overlap_thresh
        self.negpos_ratio=neg_pos
        self.variances=[0.1,0.2]
    def forward(self,predictions,targets):
        # 回归信息，置信度 先验框
        loc_data,conf_data,priors=predictions
        # 计算出batch_size
        num=loc_data.size(0)
         #取出所有的先验框
        priors=priors[:loc_data.size(1),:]
        #先验框的数量
        num_priors=(priors.size(0))
        # 创建一个tensor进行处理
        loc_t=torch.Tensor(num,num_priors,4)
        conf_t=torch.LongTensor(num,num_priors)
        for idx in range(num):
         #获得框
          truths=targets[idx][:,1:5].data
         #获得标签
          labels=targets[idx][:,0].data

         #获得先验框
          defaults=priors.data
         # 找到标签对应的先验框
          match(self.threshold, truths, defaults, labels, self.variances,loc_t,conf_t,idx)
        if torch.cuda.is_available():
             loc_t=loc_t.cuda()
             conf_t=conf_t.cuda()
         # 转化成Variable
        loc_t=Variable(loc_t,requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)
        # 所有conf_t>0的地方，代表内部包含物体
        pos = conf_t > 0





        num_pos=pos.sum(dim=1, keepdim=True)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c=loss_c.view(num,-1)
        loss_c[pos]=0
        # 获取每一张图片的softmax的结果
        _,loss_idx=loss_c.sort(1,descending=True)
        _,idx_rank=loss_idx.sort(1)
        # 计算每一张图的正样本
        num_pos = pos.long().sum(1, keepdim=True)
        # 限制负样本数量
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # 计算正样本的loss和负样本的loss
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l,loss_c