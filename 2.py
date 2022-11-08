num_epoch=10  # https://huggingface.co/openai/clip-vit-large-patch14
batch_size=5
class A():
        pass
args=A()
args.learning_rate=3e-5
args.adam_epsilon=1e-8
args.weight_decay=0


from transformers import ViTFeatureExtractor, ViTForImageClassification,ViTModel
from PIL import Image
import requests
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

from transformers import pipeline, AdamW
from transformers import AutoModelForQuestionAnswering, BertTokenizer

from PIL import Image
import requests
# 这个是最新的分类算法实现. 一个月90w的下载量. 必须掌握起来.
from transformers import CLIPProcessor, CLIPModel
from  transformers.models.clip.processing_clip import CLIPProcessorForTrain
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
processor2 = CLIPProcessorForTrain.from_pretrained("openai/clip-vit-large-patch14")

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
url = 'https://img0.baidu.com/it/u=4073830631,3465103935&fm=253&fmt=auto&app=120&f=JPEG?w=500&h=352'
url = 'https://img11.360buyimg.com/n1/jfs/t1/68853/40/18172/105212/62793f77E6ec0ede6/277fe936c4b20f24.jpg'

image = Image.open(requests.get(url, stream=True).raw)
image = Image.open('1.jpg')#人造一个假狗让他训.
#从下面一行开始debug
inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True) #文字编码,前后加上开始,和结束.  image crop_center 和normalize
image_mean= processor.current_processor.image_mean
image_std= processor.current_processor.image_std
print('mean,std',image_mean,image_std)
print('训练开始值钱进行测试.')
if 1:
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    print(probs)

#配上finetune代码=========底层是对比学习.就是给caption和image让他俩相似度最大就完事了. 只是使用的时候会提供选择题效果.

import torch
import torchvision
import numpy as np
if 1:
    #进行pil 和tensor互化:


    t=torchvision.transforms.functional.to_tensor(image)
    tt=torchvision.transforms.functional.to_pil_image(t)
    # print(image==tt)
    # print(np.asarray(image))
    # print(np.asarray(tt))
    # print(np.asarray(image)==np.asarray(tt))
    # print(1)  #证明了这2个函数是互逆的.





#=============数据部分.

from torchvision.datasets import mnist
from torchvision import datasets,transforms


import torch
from torchvision import datasets, transforms
import numpy as np

from transformers.image_utils import PILImageResampling
from transformers.image_utils import ImageFeatureExtractionMixin
class myaugmentFortrain( ImageFeatureExtractionMixin):
 
    def __init__(
 
        self,
        do_resize=True,
        size=224,
        resample=PILImageResampling.BICUBIC,
        do_center_crop=False,
        crop_size=224,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        do_convert_rgb=True,**kwargs):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.48145466, 0.4578275, 0.40821073]
        self.image_std = image_std if image_std is not None else [0.26862954, 0.26130258, 0.27577711]
        self.do_convert_rgb = do_convert_rgb
 
    def __call__(self, img):
        
        images=[img]
# transformations (convert rgb + resizing + center cropping + normalization)
        if self.do_convert_rgb:
            images = [self.convert_rgb(image) for image in images]
        if self.do_resize and self.size is not None and self.resample is not None:
            images = [
                self.resize(image=image, size=self.size, resample=self.resample, default_to_square=False)
                for image in images
            ]#进行居中crop
        if self.do_center_crop and self.crop_size is not None:
            images = [self.center_crop(image, self.crop_size) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]
        return img


transform=transforms.Compose([
                  
                   transforms.RandomHorizontalFlip(p=0.5), #按照0.5概率flip

            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.RandomResizedCrop(224,scale=(0.6, 1.0)),  # scael表示面积抽取原来的0.3

            myaugmentFortrain(),
        #    AddGaussianNoise(mean=0, variance=1, amplitude=1),
            transforms.ToTensor(), # convert a PIL image or ndarray to tensor. 
            ])


train_set = datasets.ImageFolder('smalldata_猫狗分类/train', transform=transform)
test_set = datasets.ImageFolder('smalldata_猫狗分类/test', transform=transform)


#datasets.ImageFolder源码里面说明了排序的方法.按照字母排序的.
        # classes (list): List of the class names sorted alphabetically.
        # class_to_idx (dict): Dict with items (class_name, class_index).
        # imgs (list): List of (image path, class_index) tuples






#===========锁text特征层 参数.
open_layer=-6
keys=list(model.text_model.state_dict(keep_vars=True).keys())
values=list(model.text_model.state_dict(keep_vars=True).values())
#还是锁上好!!!!!!!因为特征层已经很好了.再动会更坑.
if 1:
    for i in values[:open_layer]: #只开放最后2层.
        i.requires_grad=False
print('打印锁各个层的情况')
for name ,parm in model.text_model.state_dict(keep_vars=True).items():
    print(name,parm.requires_grad)



keys=list(model.vision_model.state_dict(keep_vars=True).keys())
values=list(model.vision_model.state_dict(keep_vars=True).values())
#还是锁上好!!!!!!!因为特征层已经很好了.再动会更坑.
if 1:
    for i in values[:open_layer]: #只开放最后2层.
        i.requires_grad=False
print('打印锁各个层的情况')
for name ,parm in model.vision_model.state_dict(keep_vars=True).items():
    print(name,parm.requires_grad)



















import torch
train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size)

import torchvision
text=["a photo of a cat", "a photo of a dog"]
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

            # all_pic=[]
            # for i in data:
            #     all_pic.append(torchvision.transforms.functional.to_pil_image(i))
            # show_tensor_pic=transforms.ToPILImage()

            all_t=[]
            for i in  target.tolist():
                all_t.append(text[i])
            

            inputs = processor(text=all_t, images=image, return_tensors="pt", padding=True) 


            #=======然后我们修改图像数据为我们提升之后的.
            data, inputs = data.to(device), inputs.to(device)

            inputs['pixel_values']=data
            inputs['return_loss']=True
            # inputs = feature_extractor(images=data, return_tensors="pt") # 这里面做了totensor和normalization
            outputs = model(**inputs)

            loss = outputs[0]
            print(loss)
            loss.backward()
            optimizer.step()

            model.zero_grad()


    print("train_over!!!!!!!!!!")

#---------下面开始测试
model.eval()
if 1:
    
    # image = Image.open(requests.get(url, stream=True).raw)
    #从下面一行开始debug
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True).to(device) #文字编码,前后加上开始,和结束.  image crop_center 和normalize
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    print(probs)



torch.save(model,'tmp2.pth') 
model=torch.load('tmp2.pth') 















