import torch
from torch.utils.data import DataLoader
from dataset import Dateset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from model import *
from MultiBoxLoss import *
import argparse
import os
import time
parser=argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
parser.add_argument("--img_size", type=int, default=448, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between saving model weights")
parser.add_argument('--epochs',type=int,default=5000,help='train epochs')
parser.add_argument('--class_nums',type=int,default=2,help='train epochs')
parser.add_argument('--save_file_location',type=str,default='./savemodel',help='save model')
#训练次数
parser.add_argument('--learn_rate',type=float,default=0.001,help='learning rate')
opt=parser.parse_args()
os.makedirs("./savemodel", exist_ok=True)
torch.backends.cudnn.benchmark = True#开启cudnn加速
cuda = True if torch.cuda.is_available() else False #判断cuda是否可用
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor#根据cuda是否可用决定tensor类型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def train():
    start_epoch=0
    batch_size=opt.batch_size#图片尺寸
    learn_rate = opt.learn_rate  # 学习率
    epochs = opt.epochs  # 训练次数
    class_nums = opt.class_nums  # 类别数
    img_size = opt.img_size#图片尺寸
    checkpoint = opt.checkpoint_interval  # 保存步数
    dataset=Dateset(batch_size=1,train_type="train")
    dataloader=DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0,collate_fn=dataset.collate_fn)
    model=get_ssd(phase="train",num_classes=class_nums)
    model.to(device)
    optimizer=optim.Adam(model.parameters(),learn_rate)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=15, verbose=True,min_lr=0.0001)
    criterion=MultiBoxloss(num_class=class_nums,overlap_thresh=0.7,neg_pos=4)
    for epoch in range(start_epoch,epochs):
        start_time=time.time()
        epoch_loss=0
        train_step=0
        loc_loss=0
        conf_loss=0
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        for batch_i, (imgs, targets) in enumerate(dataloader):
         imgs = Variable(imgs.to(device))
         targets=[Variable(label.to(device),requires_grad=False)  for label in targets]
         output=model(imgs)
         loss_l,loss_c=criterion(output, targets)
         optimizer.zero_grad()
         loss = loss_l + loss_c
         loss.backward()
         optimizer.step()
         train_step += 1
         loc_loss+=loss_l.item()
         conf_loss+=loss_c.item()
         if epoch % 10 == 0:
            torch.save(model.state_dict(), opt.save_file_location + '/model_{}'.format(epoch) + '.pth')
    scheduler.step(epoch)
    print("训练批次为{},loc_loss损失为{},conf_loss损失值为{}".format(epoch,loc_loss/train_step,conf_loss/train_step))
train()

