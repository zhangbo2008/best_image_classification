num_epoch=1
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

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
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








#=============数据部分.

from torchvision.datasets import mnist
from torchvision import datasets,transforms


import torch
from torchvision import datasets, transforms
import numpy as np
class AddGaussianNoise(torch.nn.Module):
 
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



import torch
class my_add_gaussian(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()

        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return F.hflip(img)
        return img






transform=transforms.Compose([
                  
                   transforms.RandomHorizontalFlip(p=0.5), #按照0.5概率flip

            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.RandomResizedCrop(224,scale=(0.6, 1.0)),  # scael表示面积抽取原来的0.3
        #    AddGaussianNoise(mean=0, variance=1, amplitude=1),
            transforms.ToTensor(), # convert a PIL image or ndarray to tensor. 

#===============改为tensor之后加,更好点.


            transforms.Normalize(image_mean, image_std  ) #输入均值方差.
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

            
            all_t=[]
            for i in  target.tolist():
                all_t.append(text[i])
            

            inputs = processor(text=all_t, images=image, return_tensors="pt", padding=True) 


            #=======然后我们修改图像数据为我们提升之后的.
            data, target = data.to(device), inputs.to(device)

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
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    #从下面一行开始debug
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True).to(device) #文字编码,前后加上开始,和结束.  image crop_center 和normalize
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    print(probs)



torch.save(model,'tmp2.pth') 
model=torch.load('tmp2.pth') 















