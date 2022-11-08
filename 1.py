num_epoch=100
batch_size=200
class A():
        pass
args=A()
args.learning_rate=3e-4
args.adam_epsilon=1e-8
args.weight_decay=0



from transformers import ViTFeatureExtractor, ViTForImageClassification,ViTModel
from PIL import Image
import requests
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

from transformers import pipeline, AdamW
from transformers import AutoModelForQuestionAnswering, BertTokenizer

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
url = 'https://img11.360buyimg.com/n1/jfs/t1/68853/40/18172/105212/62793f77E6ec0ede6/277fe936c4b20f24.jpg'
url = 'https://img0.baidu.com/it/u=4073830631,3465103935&fm=253&fmt=auto&app=120&f=JPEG?w=500&h=352'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224') # 这个部分只进行预处理
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224') # 这部分是神经网络加分类.

import torch.nn as nn
num_labels=2
#=============修改网络最后一层.
model.classifier= nn.Linear(768, num_labels)
model.num_labels=num_labels


#===========锁特征层 参数.

keys=list(model.state_dict(keep_vars=True).keys())
values=list(model.state_dict(keep_vars=True).values())


#还是锁上好!!!!!!!因为特征层已经很好了.再动会更坑.
if 1:
    for i in values[:-4]: #只开放最后2层.
        i.requires_grad=False






print('打印锁各个层的情况')
for name ,parm in model.state_dict(keep_vars=True).items():
    print(name,parm.requires_grad)







#=============数据部分.

from torchvision.datasets import mnist
from torchvision import datasets,transforms



from torchvision import datasets, transforms
import numpy as np
class AddGaussianNoise(object):
 
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
 
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
 
    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img
transform=transforms.Compose([
                  
                   transforms.RandomHorizontalFlip(p=0.5), #按照0.5概率flip

            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.RandomResizedCrop(224,scale=(0.6, 1.0)),  # scael表示面积抽取原来的0.3
        #    AddGaussianNoise(mean=0, variance=1, amplitude=1),
            transforms.ToTensor(), # convert a PIL image or ndarray to tensor. 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #输入均值方差.
               ])
train_set = datasets.ImageFolder('smalldata_猫狗分类/train', transform=transform)
test_set = datasets.ImageFolder('smalldata_猫狗分类/test', transform=transform)


#datasets.ImageFolder源码里面说明了排序的方法.按照字母排序的.
        # classes (list): List of the class names sorted alphabetically.
        # class_to_idx (dict): Dict with items (class_name, class_index).
        # imgs (list): List of (image path, class_index) tuples








import torch
train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size)



device='cuda'
model.to(device)
if 1:
    print('start_train')

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
   
    model.zero_grad()
    model.train()

   
    for _ in range(num_epoch):
        print('第 '+str(_+1)+' 伦')
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # inputs = feature_extractor(images=data, return_tensors="pt") # 这里面做了totensor和normalization
            outputs = model(pixel_values=data,labels=target)

            loss = outputs[0]
            print(loss)
            loss.backward()
            optimizer.step()

            model.zero_grad()


    print("train_over!!!!!!!!!!")

#---------下面开始测试
model.eval()
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(pixel_values=inputs['pixel_values'].to(device))
print(outputs.logits.argmax(-1).item())



torch.save(model,'tmp.pth') 
model=torch.load('tmp.pth') 
















































# logits = outputs.logits
# # model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])
