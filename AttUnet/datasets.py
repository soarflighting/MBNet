import torch.utils.data as data
import PIL.Image as Image
import os
import torchvision.transforms as transforms
import  albumentations as A
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

def make_dataset(train_root,mask_root):
    imgs = []
    train_filenames = os.listdir(train_root)
    labels = os.listdir(mask_root)
    for i in range(len(train_filenames)):
        img = os.path.join(train_root, train_filenames[i])
        mask = os.path.join(mask_root, labels[i])
        imgs.append((img, mask))
    return imgs


class LiverDataset(data.Dataset):
    def __init__(self, root,label,train=True,transform=True):
        imgs = make_dataset(root,label)
        self.imgs = imgs
        self.transform = transform
        self.train = train
        self.tsfm = self.get_transform()
        self.data_transform = transforms.Compose([
                            transforms.Resize((448,448),interpolation=Image.BILINEAR),
                            transforms.ToTensor(),  # -> [0,1]
                            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
        ])
        self.target_transform = transforms.Compose([
                transforms.Resize((448,448),interpolation=Image.BILINEAR),
                transforms.ToTensor()
            ])


    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.train:
            if random.random()<0.5:
                if self.transform:
                    img_x = np.asarray(img_x)
                    img_y = np.asarray(img_y)
                    image = self.tsfm(image=img_x,mask=img_y)
                    img_x = image['image']
                    img_y = image['mask']
                    img_x = Image.fromarray(img_x)
                    img_y = Image.fromarray(img_y)
        img_x = self.data_transform(img_x)
        img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

    # ????????????
    def get_transform(self):
        transform = A.Compose([
        # ????????????
        A.Rotate(),
        # Flip ??????????????????????????????
        A.Flip(always_apply=False, p=0.5),
        A.OneOf([
            # ????????????????????????????????????????????????
            A.HueSaturationValue(),
            # ??????????????????
            A.RandomBrightnessContrast(),
        ], p=1),
        # ShiftScaleRotate ???????????????????????????????????????????????????
        A.ShiftScaleRotate(),
        # ElasticTransform ????????????
        A.ElasticTransform(),
        ])

        return transform


def visual_img(imgs,labels):
    for i,img in enumerate(imgs):
        img = img.numpy().transpose(1,2,0)
        print(img.shape)
        plt.subplot(2,4,i+1)
        plt.imshow(img)
    for i,label in enumerate(labels):
        label = label.numpy().transpose(1,2,0)
        print(label.shape)
        c_dim = label.shape[-1]
        if(c_dim == 1):
            label = label.reshape(label.shape[0:2])
        plt.subplot(2,4,i+5)
        plt.imshow(label,cmap="gray")
    plt.show()




if __name__ == '__main__':
    root = "../data/train"
    label = "../data/train_label"
    liver_dataset = LiverDataset(root,label, train=True,transform=True)
    dataloaders = DataLoader(liver_dataset, batch_size=4, shuffle=True, num_workers=0)

    for img_x,img_y in dataloaders:
        visual_img(img_x,img_y)
        break




