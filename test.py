import os
import json
import argparse
import torch
import dataloaders
import models
import inspect
import math
from utils import losses
from utils import Logger
from utils.torchsummary import summary
from trainer import Trainer, check_folder_exist
import ipdb

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume, args):
    train_logger = Logger()

    # DATA LOADERS
    train_loader = get_instance(dataloaders, 'train_loader', config)
    if args.split == "val":
        val_loader = get_instance(dataloaders, 'val_loader', config)
    elif args.split == "train":
        val_loader = get_instance(dataloaders, 'train_val_loader', config)

    # MODEL
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    print(f'\n{model}\n')

    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger)

    torch.manual_seed(config['seed']['seed'])
    trainer.test()

if __name__=='__main__':
    # ipdb.set_trace()
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('--ensemble', action='store_true',
                        help="Whether to store output for ASE and deep ensemble")
    parser.add_argument("--s", default=0, type=int,
                        help="Number of deep ensemble.")
    parser.add_argument("--split", default="val", type=str, help='split for testing')
    args = parser.parse_args()

    config = json.load(open(args.config))
    config["ensemble"] = args.ensemble
    config["save_feature"]["save_feature"] = True
    if args.ensemble:
        base_path = "/data/active_testing/active_testing_seg/pytorch-segmentation/pro_data/" + config["name"] + "_ASE"
        check_folder_exist(base_path)
        config["save_feature"]["saved_path"] = os.path.join(base_path, str(args.s))
        check_folder_exist(config["save_feature"]["saved_path"])
    else:
        base_path = "/data/active_testing/active_testing_seg/pytorch-segmentation/pro_data/" + config["name"]
        check_folder_exist(base_path)
        config["save_feature"]["saved_path"] =  os.path.join(base_path, args.split)
        check_folder_exist(config["save_feature"]["saved_path"])
    # if args.resume:
    #     config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    
    main(config, args.resume, args)