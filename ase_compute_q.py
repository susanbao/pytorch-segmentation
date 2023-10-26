import typing
import io
import os

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from scipy.special import softmax
from utils.utils import *
import argparse

avgpool_32 = torch.nn.AdaptiveAvgPool2d((15,15))
avgpool_16 = torch.nn.AdaptiveAvgPool2d((30,30))
avgpool_8 = torch.nn.AdaptiveAvgPool2d((60,60))

at_image_losses = None
at_region_32_losses = None
at_region_16_losses = None
at_region_8_losses = None

ase_image_losses = None
ase_region_32_losses = None
ase_region_16_losses = None
ase_region_8_losses = None

def average_pool(array, avgpool):
    tensor = torch.from_numpy(array)
    tensor = avgpool(tensor)
    return tensor.numpy()

def con_losses(losses, loss):
    if losses is None:
        return loss
    else:
        return np.concatenate((losses, loss))
    
def compute_active_testing_losses(ego_output, dropout_output):
    global at_image_losses, at_region_32_losses, at_region_16_losses, at_region_8_losses
    pi_output = dropout_output.mean(axis=0)
    pi_output = pi_output / pi_output.sum(axis=1, keepdims=True)
    at_loss = (-1 * (pi_output * np.log(ego_output))).sum(axis=1)
    at_image_losses = con_losses(at_image_losses, np.mean(at_loss, axis=(1, 2)))
    at_region_32_losses = con_losses(at_region_32_losses, average_pool(at_loss, avgpool_32).reshape(-1))
    at_region_16_losses = con_losses(at_region_16_losses, average_pool(at_loss, avgpool_16).reshape(-1))
    at_region_8_losses = con_losses(at_region_8_losses, average_pool(at_loss, avgpool_8).reshape(-1))
    
def compute_ase_bald2_losses(ego_output, dropout_output):
    global ase_image_losses, ase_region_32_losses, ase_region_16_losses, ase_region_8_losses
    mean_output = dropout_output.mean(axis=0)
    mean_output = mean_output / mean_output.sum(axis=1, keepdims=True)
    weights = -np.log(ego_output)
    
    # sum over classes to get entropy
    entropy_average = - weights * mean_output * np.log(mean_output)
    entropy_average = entropy_average.sum(axis=1)
    
    weights = weights[np.newaxis, ...]
    average_entropy = - weights * dropout_output * np.log(dropout_output)
    average_entropy = average_entropy.sum(axis=2)
    average_entropy = average_entropy.mean(axis=0)
    
    bald = entropy_average - average_entropy
    
    ase_image_losses = con_losses(ase_image_losses, np.mean(bald, axis=(1, 2)))
    ase_region_32_losses = con_losses(ase_region_32_losses, average_pool(bald, avgpool_32).reshape(-1))
    ase_region_16_losses = con_losses(ase_region_16_losses, average_pool(bald, avgpool_16).reshape(-1))
    ase_region_8_losses = con_losses(ase_region_8_losses, average_pool(bald, avgpool_8).reshape(-1))

def main(args):
    data_type = args.model_data_type
    base_path = f"./pro_data/{data_type}_ASE/"
    output_list = [base_path + str(i) for i in range(1,5)]
    ego_output_path =  base_path + "0"
    file_list = os.listdir(ego_output_path)
    file_num = len(file_list)
    
    for file in range(file_num):
        ego_output = np_read(os.path.join(ego_output_path,f"{file}.npy"))
        ego_output = softmax(ego_output)
        dropout_output = np.expand_dims(np.copy(ego_output), axis=0)
        for output_path in output_list:
            temp_output = np_read(os.path.join(output_path,f"{file}.npy"))
            temp_output = softmax(temp_output)
            dropout_output = np.concatenate((dropout_output, np.expand_dims(np.copy(temp_output), axis=0)), axis=0)

        compute_active_testing_losses(ego_output, dropout_output)
        compute_ase_bald2_losses(ego_output, dropout_output)
        if (file+1) % 100 == 0:
            print(file)
    
    store_path = f"./ase_results/{data_type}"
    check_folder_exist(store_path)
    store_path = store_path + "/"
    np_write(at_image_losses, store_path + f"AT_{data_type}_image_Q.npy")
    np_write(at_region_32_losses, store_path + f"AT_{data_type}_region_32_Q.npy")
    np_write(at_region_16_losses, store_path + f"AT_{data_type}_region_16_Q.npy")
    np_write(at_region_8_losses, store_path + f"AT_{data_type}_region_8_Q.npy")

    np_write(ase_image_losses, store_path + f"ASE_{data_type}_image_Q.npy")
    np_write(ase_region_32_losses, store_path + f"ASE_{data_type}_region_32_Q.npy")
    np_write(ase_region_16_losses, store_path + f"ASE_{data_type}_region_16_Q.npy")
    np_write(ase_region_8_losses, store_path + f"ASE_{data_type}_region_8_Q.npy")
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_data_type', type=str, default="UNet_VOC",
                        help="mode X dataset type, current model type: PSPNet, UNet, DeepLab, FCN, SEGNet, dataset: VOC, CITY, COCO, ADE20k")
    args = parser.parse_args()
    main(args)
    print(f"Complete {args.model_data_type}")