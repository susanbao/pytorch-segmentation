import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
from utils import losses
import torch
from utils.utils import *
from scipy.special import softmax
import matplotlib.pyplot as plt
import torch.nn.functional as F

def generate_region_loss(file_losses, avgpool, region_based_losses):
    file_losses = avgpool(file_losses)
    file_losses = file_losses.numpy()
    file_losses = file_losses.reshape(-1)
    if region_based_losses is None:
        return file_losses
    else:
        return np.concatenate((region_based_losses, file_losses), axis=0)

def display_range_of_losses(losses, loss_type):
    print(f"{loss_type} loss: min  {losses.min()},  max {losses.max()}")

def run_one_split(args, split):
    base_path = "./pro_data/" + args.model_type + "/" + split + "/"
    files = os.listdir(base_path + "target")
    cross_entropy_loss_func = losses.CrossEntropyLoss2d(reduction="none")
    file_count = len(files)
    check_folder_exist(base_path + "loss")
    check_folder_exist(base_path + "entropy")
    check_folder_exist(base_path + "feature")
    count = 0
    for file in range(file_count):
        output = np_read(base_path + f"output/{file}.npy")
        target = np_read(base_path + f"target/{file}.npy")
        file_losses = cross_entropy_loss_func(torch.from_numpy(output), torch.from_numpy(target))
        file_losses = file_losses.numpy()
        np_write(file_losses, base_path + f"loss/{file}.npy")
        output = torch.from_numpy(output)
        output = F.softmax(output, dim=1)
        entropy = torch.sum(torch.mul(-output, torch.log(output + 1e-20)), dim=1).unsqueeze(dim=1)
        np_write(entropy.numpy(), base_path + f"entropy/{file}.npy")
        # np_write(output.numpy(), base_path + f"feature/{file}.npy")
        samples = output.shape[0]
        for i in range(samples):
            np_write(output[i], base_path + f"feature/{count}.npy")
            count += 1
        # if (file+1)%10 == 0:
        #     print(file)
        os.remove(base_path + f"output/{file}.npy")
            
    ## loss part
    files = os.listdir(base_path + "loss")
    file_count = len(files)
    true_image_losses = None
    avgpool_8 = torch.nn.AdaptiveAvgPool2d((60,60))
    true_region_8_losses = None
    avgpool_16 = torch.nn.AdaptiveAvgPool2d((30,30))
    true_region_16_losses = None
    avgpool_32 = torch.nn.AdaptiveAvgPool2d((15,15))
    true_region_32_losses = None
    avgpool_60 = torch.nn.AdaptiveAvgPool2d((8,8))
    true_region_60_losses = None
    for file in range(file_count):
        file_losses = np_read(base_path + f"loss/{file}.npy")
        image_losses = np.reshape(file_losses, (file_losses.shape[0], -1))
        image_losses = np.mean(image_losses, axis=1)
        if true_image_losses is None:
            true_image_losses = image_losses
        else:
            true_image_losses = np.concatenate((true_image_losses, image_losses), axis=0)
        
        file_losses = torch.from_numpy(file_losses)
        true_region_8_losses = generate_region_loss(file_losses, avgpool_8, true_region_8_losses)
        true_region_16_losses = generate_region_loss(file_losses, avgpool_16, true_region_16_losses)
        true_region_32_losses = generate_region_loss(file_losses, avgpool_32, true_region_32_losses)
        true_region_60_losses = generate_region_loss(file_losses, avgpool_60, true_region_60_losses)
        
        # if (file+1)%10 == 0:
        #     print(file)
    np_write(true_image_losses, base_path + "image_true_losses.npy")
    np_write(true_region_8_losses, base_path + "region_8_8_true_losses.npy")
    np_write(true_region_16_losses, base_path + "region_16_16_true_losses.npy")
    np_write(true_region_32_losses, base_path + "region_32_32_true_losses.npy")
    np_write(true_region_60_losses, base_path + "region_60_60_true_losses.npy")
    print(split + ":")
    display_range_of_losses(true_image_losses, "Image")
    display_range_of_losses(true_region_8_losses, "Region 8x8")
    display_range_of_losses(true_region_16_losses, "Region 16x16")
    display_range_of_losses(true_region_32_losses, "Region 32x32")
    display_range_of_losses(true_region_60_losses, "Region 60x60")

def main(args):
    run_one_split(args, 'val')
    run_one_split(args, 'train')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate data for ViT active testing')
    parser.add_argument("--model_type", default="UNet_VOC", type=str, help='model and dataset type')
    args = parser.parse_args()
    
    main(args)