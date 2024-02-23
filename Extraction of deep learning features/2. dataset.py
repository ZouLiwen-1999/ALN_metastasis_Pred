import torch.utils.data as data
import os
import PIL.Image as Image
join=os.path.join
import pandas as pd 

class MyDataset(data.Dataset):
    def __init__(self,root,split,exp_name='exp1',transform = None,target_transform = None):
        imgs=[]
        class_to_idx={}
        labels=set()

        df=pd.read_csv(join(root,split+'.csv'))  # read a table containing patient id and metastatic status
        for idx, row in df.iterrows():
            name=row['id']+'.png'
            image=join(root,name)
            label=int(int(row['N-stage'])>0)  # N-stage = 0/1
            labels.add(label)
            imgs.append([image,int(label),name])
                
        for label in labels:
            class_to_idx[label]=int(label)         
                        
        self.imgs = imgs
        self.class_to_idx=class_to_idx
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,index):
        img,label,name = self.imgs[index]
        img = Image.open(img)
        img=img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img,label,name

    def __len__(self):
        return len(self.imgs)