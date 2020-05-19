from torch.autograd import Variable
import torch
import torchvision.transforms as transform
from model import get_ssd
from PIL import Image
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class predict_SSD(object):
    def __init__(self,model_path,class_nums,confidence,confidence1,class_name=['air']):
      self.class_nums=class_nums
      self.model_path=model_path
      self.model_load()
      self.confidence=confidence
      self.class_name=class_name
      self.confidence1 = confidence1
    def model_load(self):
        model=get_ssd("test",self.class_nums)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(self.model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.net=model
        cudnn.benchmark = True
        self.net=self.net.to(device)
    def detect_img(self,image):
        img=transform.ToTensor()(image)
        image_size=img.shape[1]
        img=torch.stack([img])
        imgs = Variable(img.to(device))
        self.net.eval()
        preds=self.net(imgs)
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        top_conf=[]
        top_label=[]
        top_boxes=[]
        for i in range(preds.size(1)-1):
            j=0
            i=i+1
            print(preds[0,i,j,0])
            while preds[0,i,j,0]>=self.confidence and preds[0,i,j,0]<=self.confidence1:
                print(preds[0,i,j,0])
                score=preds[0,i,j,0]
                label_name = self.class_name[i-1]
                pt=(preds[0,i,j,1:]).detach().numpy()
                coords=[pt[0], pt[1], pt[2], pt[3]]
                top_conf.append(score)
                top_label.append(label_name)
                top_boxes.append(coords)
                j=j+1
            #将其预测结果进行解码
        print(top_conf)
        if len(top_conf)<=0:

                return  image
        top_conf = np.asarray(top_conf)
        top_label = np.array(top_label)
        top_bboxes = np.array(top_boxes)
        for i, c in enumerate(top_label):

           
           score = top_conf[i]+0.
           top, left, bottom, right = top_bboxes[i]
           top=top*image_size-2
           left=left*image_size-2
           bottom=bottom*image_size+2
           right=right*image_size+2
           top = max(0, np.floor(top + 0.5).astype('int32'))
           left = max(0, np.floor(left + 0.5).astype('int32'))
           bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
           right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))
           cv2.rectangle(img,(top,left),(bottom,right),(0,255,0))
           cv2.putText(img,str(score),(top-5,left-5),cv2.FONT_HERSHEY_DUPLEX, 1, (100, 200, 200), 1)
        cv2.imshow('img',img)
        cv2.waitKey(0)

pre=predict_SSD("./savemodel/model_200.pth",2,0.8,1)
img=Image.open('./dataset/test/img/6.png').convert('RGB')
pre.detect_img(img)

