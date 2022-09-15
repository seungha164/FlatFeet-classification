import cv2,os
import numpy as np
from torch.utils.data import Dataset
import torch

# 1. Single Input
class SingleDataset(Dataset):
    def __init__(self, img_list, class_to_int, transforms = None):
        super().__init__()
        self.img_list = img_list          # 이미지 경로 리스트   
        self.class_to_int = class_to_int    # class_to_int = {'normal':0,'flatfeet':1}
        self.transforms = transforms        # 이미지 전처리를 위한 torchvision.transform
    
    def getImg(self,x):
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32)
        x /= 255.0
        return x

    def __getitem__(self, index):

        img_path = self.img_list[index]
        
        #Reading image
        image = self.getImg(img_path)
        
        #Retriving class label
        label = img_path.split("/")[-2]
        label = self.class_to_int[label]
        
        #Applying transforms on image
        if self.transforms:
            image = self.transforms(image)

        return image,label
          
    def __len__(self):
        return len(self.img_list)

# 2. Dual INPUT 
class DualDataset(Dataset): 
    def __init__(self, imgL_list, imgR_list, class_to_int, transforms = None):
        super().__init__()
        self.imgL_list = imgL_list          # 이미지 경로 리스트
        self.imgR_list = imgR_list      
        self.class_to_int = class_to_int    # class_to_int = {'normal':0,'flatfeet':1}
        self.transforms = transforms        # 이미지 전처리를 위한 torchvision.transform
    
    def getImg(self, x):
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32)
        x /= 255.0
        return x

    def __getitem__(self, index):
        img = None
        imgL_path = self.imgL_list[index]
        imgR_path = self.imgR_list[index]

        #Reading image
        imageL = self.getImg(imgL_path)
        imageR = self.getImg(imgR_path)
        
        #Retriving class label
        label = imgL_path.split("/")[-3]
        label = self.class_to_int[label]
        
        #Applying transforms on image
        if self.transforms:
            imageL = self.transforms(imageL)
            imageR = self.transforms(imageR)
        
        return imageL, imageR, label

    def __len__(self):
        return len(self.imgL_list)

# 3. Triple input
class TripleDataset(Dataset): 
    def __init__(self, imgL_list, imgR_list, imgF_list, class_to_int, transforms = None):
        super().__init__()
        self.imgL_list = imgL_list          # 이미지 경로 리스트
        self.imgR_list = imgR_list      
        self.imgF_list = imgF_list 
        self.class_to_int = class_to_int    # class_to_int = {'normal':0,'flatfeet':1}
        self.transforms = transforms        # 이미지 전처리를 위한 torchvision.transform
    
    def getImg(self, x):
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32)
        x /= 255.0
        return x

    def __getitem__(self, index):
        img = None
        imgL_path = self.imgL_list[index]
        imgR_path = self.imgR_list[index]
        imgF_path = self.imgF_list[index]

        #Reading image
        imageL = self.getImg(imgL_path)
        imageR = self.getImg(imgR_path)
        imageF = self.getImg(imgF_path)
        
        
        #Retriving class label
        label = imgL_path.split("/")[-3]
        label = self.class_to_int[label]
        
        #Applying transforms on image
        if self.transforms:
            imageL = self.transforms(imageL)
            imageR = self.transforms(imageR)
            imageF = self.transforms(imageF)
        
        return imageL, imageR, imageF, label
        
    def __len__(self):
        return len(self.imgL_list)

# + channel 6개
class FeetDatasetC6(Dataset): 
    def __init__(self, imgL_list, imgR_list, class_to_int, transforms = None):
        super().__init__()
        self.imgL_list = imgL_list          # 이미지 경로 리스트
        self.imgR_list = imgR_list      
        self.class_to_int = class_to_int    # class_to_int = {'normal':0,'flatfeet':1}
        self.transforms = transforms        # 이미지 전처리를 위한 torchvision.transform
    
    def getImg(self,x,is_flip):
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        if is_flip:
            x = cv2.flip(x,1)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32)
        x /= 255.0
        return x

    def __getitem__(self, index):
        img = None
        imgL_path = self.imgL_list[index]
        imgR_path = self.imgR_list[index]

        #Reading image
        imageL = self.getImg(imgL_path, False)
        imageR = self.getImg(imgR_path, True)
        
        #Retriving class label
        label = imgL_path.split("/")[-3]
        label = self.class_to_int[label]
        
        #Applying transforms on image
        if self.transforms:
            imageL = self.transforms(imageL)
            imageR = self.transforms(imageR)
            #print(imageL.size())
            img = torch.cat([imageL,imageR],0)

        return img,label
        
    def __len__(self):
        return len(self.imgL_list)

def getDataset(datasetN, _path, _transform):
    ## 1. path 저장
    labels = {'normal':0, 'flatfeet':1}
    if datasetN == 'SingleDataset':
        imgCs =[]
        for _class in ['normal/','flatfeet/']:
            for img in os.listdir(_path + _class):
                imgCs.append(_path + _class+img)
        return SingleDataset(imgCs, labels, _transform)

    imgLs, imgRs, imgFs = [],[],[]
    for _class in ['normal','flatfeet']:
        for img in os.listdir(_path + _class+'/L'):
            imgLs.append(_path + _class+'/L' + "/" + img)
            imgRs.append(_path + _class+'/R' + "/" + img)
            imgFs.append(_path + _class+'/F' + "/" + img)

    ## 2. dataset에 맞춰 return
    if datasetN=='FeetDatasetC6':
        return FeetDatasetC6(imgLs, imgRs, labels, _transform)
    elif datasetN=='DualDataset':
        return DualDataset(imgLs, imgRs, labels, _transform)
    elif datasetN=='TripleDataset':
        return TripleDataset(imgLs,imgRs,imgFs, labels, _transform)