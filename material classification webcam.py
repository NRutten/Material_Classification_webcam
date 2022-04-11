# -*- coding: utf-8 -*-
"""
Material classification, take a photo with a webcam and identify the texture in the photo.

Created on Wed Mar 16 13:46:30 2022

@author: Niek Rutten
"""

## Python standard libraries
from __future__ import print_function
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
import os
import cv2
from sklearn.metrics import matthews_corrcoef
import pickle
import keyboard
from time import sleep
from pyfirmata import Arduino, SERVO, util

# variables for Arduino
port = 'COM7'
pin1 = 8
pin2= 9
pin3 = 10
pin4= 11

board = Arduino(port)

board.digital[pin1].mode=SERVO
board.digital[pin1].write(90) #start of by having servo stand still
board.digital[pin2].mode=SERVO
board.digital[pin2].write(90) #start of by having servo stand still
board.digital[pin3].mode=SERVO
board.digital[pin3].write(90) #start of by having servo stand still
board.digital[pin4].mode=SERVO
board.digital[pin4].write(90) #start of by having servo stand still

from barbar import Bar
from Prepare_Data import Prepare_Webcam_DataLoaders


from Demo_Parameters import Network_parameters
from Utils.Network_functions import initialize_model, train_model,test_model

## PyTorch dependencies
import torch
import torch.nn as nn

#Array of texture categories from DTD
categories = ["banded", "blotchy", "braided", "bubbly", "bumpy",
              "chequered","cobwebbed","cracked","crosshatched", "crystalline",
              "dotted", "fibrous", "flecked", "freckled", "frilly", "gauzy",
              "grid", "grooved", "honeycombed", "interlaced","knitted",
              "lacelike", "lined","marbled", "matted", "meshed", "paisley","perforated",
              "pitted", "pleated", "polka-dotted", "porous", "potholed",
              "scaly","smeared","spiralled", "sprinkled",
              "stained","stratified","striped",
              "studded","swirly","veined","waffled","woven","wrinkled","zigzagged"]


#servos mapped to the texture categories
texture_Output = [3,4,2,2,2,3,2,1,2,3,3,1,2,3,4,3,3,2,4,3,2,2,3,3,4,1,3,4,4,2,3,1,4,1,2,3,4,4,1,3,1,3,4,2,2,2,3]


## Local external libraries
from Utils.Generate_TSNE_visual import Generate_TSNE_visual
from Texture_information import Class_names
from Demo_Parameters import Network_parameters as Results_parameters
from Utils.Network_functions import initialize_model
from Prepare_Data_Results import Prepare_DataLoaders
from Utils.RBFHistogramPooling import HistogramLayer as RBFHist
from Utils.LinearHistogramPooling import HistogramLayer as LinearHist
from Utils.Confusion_mats import plot_confusion_matrix,plot_avg_confusion_matrix

#Location of experimental results
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fig_size = Results_parameters['fig_size']
font_size = Results_parameters['font_size']

NumRuns = Results_parameters['Splits'][Results_parameters['Dataset']]
plot_name = Results_parameters['Dataset'] + ' Test Confusion Matrix'
avg_plot_name = Results_parameters['Dataset'] + ' Test Average Confusion Matrix'
class_names = Class_names[Results_parameters['Dataset']]
cm_stack = np.zeros((len(class_names),len(class_names)))
cm_stats = np.zeros((len(class_names),len(class_names),NumRuns))
FDR_scores = np.zeros((len(class_names),NumRuns))
log_FDR_scores = np.zeros((len(class_names),NumRuns))
accuracy = np.zeros(NumRuns)
MCC = np.zeros(NumRuns)

#Name of dataset
Dataset = Results_parameters['Dataset']

#Model(s) to be used
model_name = Results_parameters['Model_names'][Dataset]

#Number of classes in dataset
num_classes = Results_parameters['num_classes'][Dataset]

#Number of runs and/or splits for dataset
numRuns = Results_parameters['Splits'][Dataset]

#Number of bins and input convolution feature maps after channel-wise pooling
numBins = Results_parameters['numBins']
num_feature_maps = Results_parameters['out_channels'][model_name]

#Local area of feature map after histogram layer
feat_map_size = Results_parameters['feat_map_size']

#function to run prediction
def feed_data(dataloader,model,device):
    #Initialize and accumalate ground truth, predictions, and image indices
    model.eval()

    # Iterate over data
    print('running prediction')
    with torch.no_grad():
        for idx, (inputs, labels,index) in enumerate(Bar(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            index = index.to(device)

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            print(categories[preds[0].item()]);
    return preds

for split in range(0, NumRuns):

    #Set directory location for models
    if(Results_parameters['histogram']):
        if(Results_parameters['parallel']):
            sub_dir = (Results_parameters['folder'] + Results_parameters['mode']
                        + '/' + Results_parameters['Dataset'] + '/'
                        + Results_parameters['hist_model']  + '_'+
                        Results_parameters['histogram_type'] + '/Parallel/Run_'
                        + str(split + 1) + '/')
        else:
            sub_dir = (Results_parameters['folder'] + Results_parameters['mode']
                        + '/' + Results_parameters['Dataset'] + '/'
                        + Results_parameters['hist_model']  + '_'+
                        Results_parameters['histogram_type'] + '/Inline/Run_'
                        + str(split + 1) + '/')
    #Baseline model
    else:
        sub_dir = (Results_parameters['folder'] + Results_parameters['mode']
                    + '/' + Results_parameters['Dataset'] + '/GAP_' +
                    Results_parameters['Model_names'][Results_parameters['Dataset']]
                    + '/Run_' + str(split + 1) + '/')

    #Load model
    if Results_parameters['histogram_type'] == 'RBF':
        histogram_layer = RBFHist(int(num_feature_maps/(feat_map_size*numBins)),
                                  Results_parameters['kernel_size'][model_name],
                                  num_bins=Results_parameters['numBins'],stride=Results_parameters['stride'],
                                  normalize_count=Results_parameters['normalize_count'],
                                  normalize_bins=Results_parameters['normalize_bins'])
    elif Results_parameters['histogram_type'] == 'Linear':
        histogram_layer = LinearHist(int(num_feature_maps/(feat_map_size*numBins)),
                                  Results_parameters['kernel_size'][model_name],
                                  num_bins=Results_parameters['numBins'],stride=Results_parameters['stride'],
                                  normalize_count=Results_parameters['normalize_count'],
                                  normalize_bins=Results_parameters['normalize_bins'])
    else:
        raise RuntimeError('Invalid type for histogram layer')

    # Initialize the histogram model for this run
    model, input_size = initialize_model(model_name, num_classes,
                                            Results_parameters['in_channels'][model_name],
                                            num_feature_maps,
                                            feature_extract = Results_parameters['feature_extraction'],
                                            histogram= Results_parameters['histogram'],
                                            histogram_layer=histogram_layer,
                                            parallel=Results_parameters['parallel'],
                                            use_pretrained=Results_parameters['use_pretrained'],
                                            add_bn = Results_parameters['add_bn'],
                                            scale = Results_parameters['scale'],
                                            feat_map_size=feat_map_size)

    device_loc = torch.device(device)
    best_weights = torch.load(sub_dir + 'Best_Weights.pt',map_location=device_loc)
    #If parallelized, need to set change model
    if Results_parameters['Parallelize']:
        model = nn.DataParallel(model)
    model.load_state_dict(best_weights)
    model = model.to(device)

while True:  # making a loop
    try:  # used try so that if user pressed other than the given key error will not be shown
        if keyboard.is_pressed(' '):  # if space is pressed take picture and run prediction
            sleep(3)
            dataloaders_dict = Prepare_Webcam_DataLoaders(Network_parameters,split,input_size=input_size)
            test_dict = feed_data(dataloaders_dict['test'],model,device)
            x = texture_Output[test_dict[0].item()] #check which texture is detected and move servo accordingly
            if x == 4:
                print("output is going ")
                board.digital[pin1].write(0) #rotate clockwise
                sleep(0.6)
                board.digital[pin1].write(90)
                sleep(3)
                board.digital[pin1].write(180) #rotate clockwise
                sleep(0.6)
                board.digital[pin1].write(90)
                sleep(0.01)

            elif x == 3:
                board.digital[pin2].write(0) #rotate clockwise
                sleep(0.5)
                board.digital[pin2].write(90)
                sleep(3)
                board.digital[pin2].write(180) #rotate clockwise
                sleep(0.5)
                board.digital[pin2].write(90)
                sleep(3)

            elif x == 2:
                board.digital[pin3].write(0) #rotate clockwise
                sleep(0.4)
                board.digital[pin3].write(90)
                sleep(3)
                board.digital[pin3].write(180) #rotate clockwise
                sleep(0.4)
                board.digital[pin3].write(90)
                sleep(3)

            elif x ==  1:
                board.digital[pin4].write(0) #rotate clockwise
                sleep(0.5)
                board.digital[pin4].write(90)
                sleep(3)
                board.digital[pin4].write(180) #rotate clockwise
                sleep(0.5)
                board.digital[pin4].write(90)
                sleep(0.01)
        if keyboard.is_pressed('Esc'):  # if escape key is pressed exit while loop and program
            print("leaving")
            break
    except:
        break