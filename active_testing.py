import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
import torch
import pickle
import copy
import random
from utils.utils import *

# random_seed_set = [4519, 9524, 5901, 1028, 6382, 5383, 5095, 7635,  890,  608]
random_seed_set = [4519, 9524, 5901]

def LURE_weights_for_risk_estimator(weights, N):
    M = weights.size
    if M < N:
        m = np.arange(1, M+1)
        v = (
            1
            + (N-M)/(N-m) * (
                    1 / ((N-m+1) * weights)
                    - 1
                    )
            )
    else:
        v = 1

    return v

def acquire(expected_loss_inputs, samples_num):
    assert samples_num <= expected_loss_inputs.size
    expected_loss = np.copy(expected_loss_inputs)
    # Log-lik can be negative.
    # Make all values positive.
    if (expected_loss < 0).sum() > 0:
        expected_loss += np.abs(expected_loss.min())
    
    if np.any(np.isnan(expected_loss)):
        print('Found NaN values in expected loss, replacing with 0.')
        # print(f'{expected_loss}')
        expected_loss = np.nan_to_num(expected_loss, nan=0)
    pick_sample_idxs = np.zeros((samples_num), dtype = int)
    idx_array = np.arange(expected_loss.size)
    weights = np.zeros((samples_num), dtype = np.single)
    uniform_clip_val = 0.2
    expected_loss = np.asarray(expected_loss).astype('float64')
    for i in range(samples_num):
        expected_loss /= expected_loss.sum()
        # clip all values less than 10 percent of uniform propability
        expected_loss = np.maximum(uniform_clip_val * 1/expected_loss.size, expected_loss)
        expected_loss /= expected_loss.sum()
        sample = np.random.multinomial(1, expected_loss)
        cur_idx = np.where(sample)[0][0]
        # cur_idx = np.random.randint(expected_loss.size)
        pick_sample_idxs[i] = idx_array[cur_idx]
        weights[i] = expected_loss[cur_idx]
        selected_mask = np.ones((expected_loss.size), dtype=bool)
        selected_mask[cur_idx] = False
        expected_loss = expected_loss[selected_mask]
        idx_array = idx_array[selected_mask]
    return pick_sample_idxs, weights

def run_one_random_sample_risk_estimator(true_losses, seed, samples_num):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    perm = np.random.permutation(true_losses.size)
    pick_sample_idxs = perm[:samples_num]
    sampled_true_losses = true_losses[pick_sample_idxs]
    return float(sampled_true_losses.mean())

def run_one_active_test_risk_estimator(true_losses, expected_losses, seed, samples_num):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pick_sample_idxs, weights = acquire(expected_losses, samples_num)
    risk_estimator_weights = LURE_weights_for_risk_estimator(weights, expected_losses.size)
    sampled_true_losses = true_losses[pick_sample_idxs]

    loss_risk = (sampled_true_losses * risk_estimator_weights).mean()
    return float(loss_risk)

# def active_testing(file_path, true_losses, expected_losses, active_test_type, sample_size_set, display = False):
#     json_object = {}
#     for sample_size in sample_size_set:
#         for seed in random_seed_set:
#             result = {"active_test_type": active_test_type, "sample_size": sample_size}
#             loss_risk = run_one_active_test_risk_estimator(true_losses, expected_losses, seed, sample_size)
#             result["loss"] = loss_risk
#             json_object[len(json_object)] = result
#         if display:
#             print(f"Complete simple size : {sample_size}")
#     with open(file_path, "w") as outfile:
#         json.dump(json_object, outfile)
        
def active_testing(file_path, true_losses, expected_losses, active_test_type, sample_size_set, display = False):
    json_object = {}
    for seed in random_seed_set:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pick_sample_idxs, weights = acquire(expected_losses, sample_size_set[-1])
        for sample_size in sample_size_set:
            result = {"active_test_type": active_test_type, "sample_size": sample_size}
            risk_estimator_weights = LURE_weights_for_risk_estimator(weights[:sample_size], expected_losses.size)
            sampled_true_losses = true_losses[pick_sample_idxs[:sample_size]]
            loss_risk = (sampled_true_losses * risk_estimator_weights).mean()
            result["loss"] = loss_risk
            json_object[len(json_object)] = result
        if display:
            print(f"Complete seed : {seed}")
    with open(file_path, "w") as outfile:
        json.dump(json_object, outfile)
        
def get_whole_data_set_risk_estimator(true_losses):
    return float(true_losses.mean())

def main(args):
    split = args.split
    # PSPNet_VOC, UNet_COCO10k, UNet_VOC, DeepLab_VOC, FCN_VOC, SEGNet_VOC, PSPNet_CITY
    model_dataset = args.model_data_type
    if model_dataset[-1] == "0":
        model_origin_folder = model_dataset[:-3]
    else:
        model_origin_folder = model_dataset
    base_path = f"./pro_data/{model_origin_folder}/{split}/"
    data_type = args.data_type # image, region_8, region_16
    store_main_folder = "runs_3_large_range" # runs_3, runs_3_large_range, runs_10_large_range
    check_folder_exist(f"./results/{store_main_folder}/{model_dataset}")
    
    step_list = {5000,10000,15000,20000}
    # step_list = {25000}
    # step_list = {35000}
    # step_list = {25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000}
    # step_list = {25000,30000, 35000, 40000, 45000,55000, 65000, 70000, 75000, 80000,85000,90000,95000, 100000}
    # step_list = np.arange(25000, 60001, 5000)
    # step_list = {65000, 70000, 75000, 80000,85000,90000,95000, 100000}
    
    baseline_run = True
    
    if data_type == "image":
        true_losses = np_read(base_path + "image_true_losses.npy")
        # sample_size_precentage = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045,
        #                       0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08]
        if store_main_folder == "runs_3":
            sample_size_precentage = np.linspace(0.01, 0.08, 20)
        else:
            if model_origin_folder[-4:] == "CITY":
                sample_size_precentage = np.linspace(0.001, 0.5, 500)
                sample_size_precentage = sample_size_precentage[2:]
            else:
                sample_size_precentage = np.linspace(0.001, 0.5, 500)
        result_json_path = f"./results/{store_main_folder}/{model_dataset}/image_based_active_testing/"
        vit_base_path = "../ViT-pytorch/output/"
    if data_type == "image_2":
        true_losses = np_read(base_path + "image_split_2_2_true_losses.npy")
        sample_size_precentage = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045,
                              0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08]
        result_json_path = f"./results/{store_main_folder}/{model_dataset}/image_split_2_2_active_testing/"
        vit_base_path = "../ViT-pytorch/output/"
    elif data_type == "region_8":
        true_losses = np_read(base_path + "region_8_8_true_losses.npy")
        sample_size_precentage = np.linspace(0.00001, 0.0001, 20)
        result_json_path = f"./results/{store_main_folder}/{model_dataset}/region_8_8_active_testing/"
        # result_json_path = "./results/region_8_8_try/"
        vit_base_path = "../ViT-pytorch/output/region_8_8/"
    elif data_type == "region_16":
        true_losses = np_read(base_path + "region_16_16_true_losses.npy")
        # sample_size_precentage = np.linspace(0.0001, 0.001, 20)
        if store_main_folder == "runs_3":
            sample_size_precentage = np.linspace(0.0002, 0.001, 20)
        else:
            sample_size_precentage = np.linspace(0.00001, 0.01, 1000)
        result_json_path = f"./results/{store_main_folder}/{model_dataset}/region_16_16_active_testing/"
        # vit_base_path = "../ViT-pytorch/output/region_16_16/"
        vit_base_path = "../ViT-pytorch/output/"
    elif data_type == "region_60":
        true_losses = np_read(base_path + "region_60_60_true_losses.npy")
        sample_size_precentage = np.linspace(0.0001, 0.001, 20)
        result_json_path = f"./results/{store_main_folder}/{model_dataset}/region_60_60_active_testing/"
        vit_base_path = "../ViT-pytorch/output/"
    elif data_type == "region_30":
        true_losses = np_read(base_path + "region_30_30_true_losses.npy")
        sample_size_precentage = np.linspace(0.0001, 0.001, 20)
        result_json_path = f"./results/{store_main_folder}/{model_dataset}/region_30_30_active_testing/"
        vit_base_path = "../ViT-pytorch/output/"
    elif data_type == "region_32":
        true_losses = np_read(base_path + "region_32_32_true_losses.npy")
        # sample_size_precentage = np.linspace(0.0001, 0.001, 20)
        if store_main_folder == "runs_3":
            sample_size_precentage = np.linspace(0.0002, 0.001, 20)
        else:
            sample_size_precentage = np.linspace(0.00001, 0.01, 1000)
        result_json_path = f"./results/{store_main_folder}/{model_dataset}/region_32_32_active_testing/"
        vit_base_path = "../ViT-pytorch/output/"

    check_folder_exist(result_json_path)
    box_labels_nums = true_losses.shape[0]
    sample_size_set = (np.array(sample_size_precentage) * box_labels_nums).astype(int).tolist()
    
    if baseline_run:
        ## random sample
        file_path = result_json_path + "random_sample_3_runs.json"
        json_object = {}
        for sample_size in sample_size_set:
            for seed in random_seed_set:
                result = {"active_test_type": "random sample", "sample_size": sample_size}
                loss_risk = run_one_random_sample_risk_estimator(true_losses, seed, sample_size)
                result["loss"] = float(loss_risk)
                json_object[len(json_object)] = result
        write_one_results(json_object, file_path)

        # whole dataset
        file_path = result_json_path + "None.json"
        result = {"active_test_type": "None", "sample_size": true_losses.size}
        result["loss"] = get_whole_data_set_risk_estimator(true_losses)
        json_object = {}
        json_object[0] = result
        with open(file_path, "w") as outfile:
            json.dump(json_object, outfile)
    
    
    ase_store_path = f"./ase_results/{model_origin_folder}/"
    if data_type == "image":
        for train_step in step_list:
            val_estimated_loss = np.array(read_one_results(f"../ViT-pytorch/output/ViT_{model_dataset}_all_losses_{train_step}.json")['losses'])
            file_path = result_json_path + f"ViT_all_runs_{train_step}.json"
            active_testing(file_path, true_losses, val_estimated_loss, "ViT all", sample_size_set)
        if baseline_run:
            at_losses = np_read(ase_store_path + f"AT_{model_origin_folder}_image_Q.npy")
            file_path = result_json_path + f"AT_runs.json"
            active_testing(file_path, true_losses, at_losses, "AT image", sample_size_set)

            ase_losses = np_read(ase_store_path + f"ASE_{model_origin_folder}_image_Q.npy")
            file_path = result_json_path + f"ASE_runs.json"
            active_testing(file_path, true_losses, ase_losses, "ASE image", sample_size_set)
    elif data_type == "region_16":
        for train_step in step_list:
            expected_losses = np.array(read_one_results(vit_base_path + f"ViT_{model_dataset}_region_losses_{train_step}.json")['losses']).squeeze()
            # expected_losses = np.array(read_one_results(vit_base_path + f"ViT_{model_dataset}_all_region_losses_{train_step}.json")['losses']).squeeze()
            file_path = result_json_path + f"ViT_region_runs_{train_step}.json"
            active_testing(file_path, true_losses, expected_losses, "ViT region", sample_size_set, display=False)
        if baseline_run:    
            at_losses = np_read(ase_store_path + f"AT_{model_origin_folder}_region_16_Q.npy")
            file_path = result_json_path + f"AT_runs.json"
            active_testing(file_path, true_losses, at_losses, "AT region", sample_size_set)
            ase_losses = np_read(ase_store_path + f"ASE_{model_origin_folder}_region_16_Q.npy")
            file_path = result_json_path + f"ASE_runs.json"
            active_testing(file_path, true_losses, ase_losses, "ASE region", sample_size_set)
    elif data_type == "region_32":
        for train_step in step_list:
            expected_losses = np.array(read_one_results(vit_base_path + f"ViT_{model_dataset}_region_32_32_losses_{train_step}.json")['losses']).squeeze()
            file_path = result_json_path + f"ViT_region_runs_{train_step}.json"
            active_testing(file_path, true_losses, expected_losses, "ViT region 32", sample_size_set, display=False)
        
        if baseline_run:
            at_losses = np_read(ase_store_path + f"AT_{model_origin_folder}_region_32_Q.npy")
            file_path = result_json_path + f"AT_runs.json"
            active_testing(file_path, true_losses, at_losses, "AT region 32", sample_size_set)

            ase_losses = np_read(ase_store_path + f"ASE_{model_origin_folder}_region_32_Q.npy")
            file_path = result_json_path + f"ASE_runs.json"
            active_testing(file_path, true_losses, ase_losses, "ASE region 32", sample_size_set)
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_data_type', type=str, default="UNet_VOC",
                        help="mode X dataset type, current model type: PSPNet, UNet, DeepLab, FCN, SEGNet, dataset: VOC, CITY, COCO, ADE20k")
    parser.add_argument("--data_type", type=str, default="image",
                        help="Region or image.")
    parser.add_argument("--split", default="val", type=str,
                        help="val/train")
    args = parser.parse_args()
    main(args)
    print(f"Complete {args.model_data_type} {args.data_type}")