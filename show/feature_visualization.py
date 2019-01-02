# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import models

import tkinter.filedialog
import matplotlib.pyplot as plt
from PIL import Image

from load import loadnet

import random

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
std = (0.2023, 0.1994, 0.2010)
mean = (0.4914, 0.4822, 0.4465)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def process_image(fname):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    ])

    img = Image.open(fname).convert('RGB')
    plt.imshow(img)
    plt.show()
    img = img.resize((32, 32), Image.ANTIALIAS)
    plt.imshow(img)
    plt.show()
    inputs = transform(img).reshape(1, 3, 32, 32)
    return inputs

class FeatureVisualization():
    def __init__(self,inputs,selected_layer,pretrained_model):
        self.inputs=inputs
        self.selected_layer=selected_layer
        self.pretrained_model = pretrained_model.module.features
        #self.pretrained_model = models.vgg16(pretrained=True).features

    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input=self.inputs
        #print(input)
        #print(input.shape)
        x=input
        for index,layer in enumerate(self.pretrained_model):
            x=layer(x)
            #print(index,x.size())
            if (index == self.selected_layer):
                return x

    def get_single_feature(self,feature_index):
        features=self.get_feature()
        #print(features.shape)

        feature=features[:,feature_index,:,:]
        #print(feature.shape)

        feature=feature.view(feature.shape[1],feature.shape[2])
        print(feature.shape)

        return feature

    def get_some_feature(self):

        features=self.get_feature()
        #print(features.shape)
#        rand_index = []
#        for i in range(6):
#            rand_index.append(random.randint(0,features.shape[1]))
#        rand_index.sort()

        feature = features[:,0:12,:,:]

        #print(feature.shape)

        feature=feature.view(12,feature.shape[2],feature.shape[3])
        
        print(feature[0].shape)

        return feature

    def show_feature(self):
        #to numpy
        #feature=self.get_single_feature(0)
        feature=self.get_some_feature()
        feature=feature.data.cpu().numpy()

        #print(feature[0])
        #use sigmod to [0,1]
        feature= 1.0/(1+np.exp(-1*feature))
        #print(feature[0])
        # to [0,255]
        feature=np.round(feature*255)

        #cv2.imwrite('./img.jpg',feature)
        plt.subplot(341)
        plt.imshow(feature[0],cmap='gray')
        plt.subplot(342)
        plt.imshow(feature[1],cmap='gray')
        plt.subplot(343)
        plt.imshow(feature[2],cmap='gray')
        plt.subplot(344)
        plt.imshow(feature[3],cmap='gray')
        plt.subplot(345)
        plt.imshow(feature[4],cmap='gray')
        plt.subplot(346)
        plt.imshow(feature[5],cmap='gray')
        plt.subplot(347)
        plt.imshow(feature[6],cmap='gray')
        plt.subplot(348)
        plt.imshow(feature[7],cmap='gray')
        plt.subplot(349)
        plt.imshow(feature[8],cmap='gray')
        plt.subplot(3,4,10)
        plt.imshow(feature[9],cmap='gray')
        plt.subplot(3,4,11)
        plt.imshow(feature[10],cmap='gray')
        plt.subplot(3,4,12)
        plt.imshow(feature[11],cmap='gray')
        plt.show()

if __name__ == '__main__':
    # 原因: 如 https://blog.csdn.net/xiemanR/article/details/71700531
    # test loader涉及多线程操作, 在windows环境下需要用__name__ == '__main__'包装

    net, _ = loadnet(7)
    net = net.to(device)
    net.eval()  # 变为测试模式, 对dropout和batch normalization有影响

    fname = tkinter.filedialog.askopenfilename()
    inputs = process_image(fname).to(device)

    print(net.module)

    selected_layer = int(input("输入你选择的层数（0~44,其他输入程序结束）:"))
    while 0<=selected_layer<=44:
        myClass=FeatureVisualization(inputs,selected_layer,net)
        myClass.show_feature()
        selected_layer = int(input("输入你选择的层数（0~44,其他输入程序结束）:"))
    # get class