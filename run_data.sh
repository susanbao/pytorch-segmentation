# CUDA_VISIBLE_DEVICES=3 python test.py --config ./config/UNet_VOC.json --resume saved/UNetResnet_VOC/09-21_15-36/checkpoint-epoch50.pth
cuda=$1
config=$2

CUDA_VISIBLE_DEVICES=$cuda python test.py --split val --config ./config/$config.json --resume saved/$config/best/checkpoint-epoch30.pth
CUDA_VISIBLE_DEVICES=$cuda python test.py --split train --config ./config/$config.json --resume saved/$config/best/checkpoint-epoch30.pth

CUDA_VISIBLE_DEVICES=$cuda python generate_data.py --model_type $config

rm -rf ./pro_data/$config/train/output
rm -rf ./pro_data/$config/val/output