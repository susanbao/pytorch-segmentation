cuda=$1
data_type=$2

CUDA_VISIBLE_DEVICES=$cuda python test.py --config ./config/$data_type.json --resume saved/$data_type/best/best_model.pth --split val --ensemble --s 0
CUDA_VISIBLE_DEVICES=$cuda python test.py --config ./config/$data_type.json --resume saved/$data_type/best/checkpoint-epoch50.pth --split val --ensemble --s 1
CUDA_VISIBLE_DEVICES=$cuda python test.py --config ./config/$data_type.json --resume saved/$data_type/best/checkpoint-epoch60.pth --split val --ensemble --s 2
CUDA_VISIBLE_DEVICES=$cuda python test.py --config ./config/$data_type.json --resume saved/$data_type/best/checkpoint-epoch70.pth --split val --ensemble --s 3
CUDA_VISIBLE_DEVICES=$cuda python test.py --config ./config/$data_type.json --resume saved/$data_type/best/checkpoint-epoch80.pth --split val --ensemble --s 4

python ase_compute_q.py --model_data_type $data_type

rm -rf ./pro_data/${data_type}_ASE