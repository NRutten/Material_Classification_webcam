# -*- coding: utf-8 -*-
"""
Loading an image from the webcam into a dataloader for the algorithm to analyze.

Created on Mon Mar 21 11:41:15 2022

@author: Niek Rutten
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision
import pdb
import torch
import matplotlib.pyplot as plt
import statistics
import numpy as np
import time
import cv2
from torchvision.utils import save_image


class webcam_data():
    def __init__(self, img_transform=None):
        self.img_transform = img_transform

    def __len__(self):
        return 1

    def __getitem__(self, index):

        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) #open camera

        returnValue, image = camera.read() #capture image

        colorIMG = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#convert to correect type
        img = Image.fromarray(colorIMG)
        #img.save('img.jpg')
        label = torch.tensor(0)

        if self.img_transform is not None:
            img = self.img_transform(img)

        del(camera)
        cv2.destroyAllWindows()
        return img, label,index #return image


##    if Dataset=='DTD':
##        train_dataset = DTD_data(data_dir, data='train',
##                                           numset = split + 1,
 ##                                          img_transform=data_transforms['train'])

