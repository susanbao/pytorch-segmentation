# python active_testing.py --model_data_type UNet_VOC --data_type image
# python active_testing.py --model_data_type PSPNet_VOC --data_type image
# python active_testing.py --model_data_type DeepLab_VOC --data_type image
python active_testing.py --model_data_type FCN_VOC --data_type image
python active_testing.py --model_data_type SEGNet_VOC --data_type image

python active_testing.py --model_data_type PSPNet_CITY --data_type image
python active_testing.py --model_data_type UNet_CITY --data_type image
python active_testing.py --model_data_type SEGNet_CITY --data_type image
# python active_testing.py --model_data_type DeepLab_CITY --data_type image
python active_testing.py --model_data_type FCN_CITY --data_type image

python active_testing.py --model_data_type UNet_VOC --data_type region_32
# python active_testing.py --model_data_type PSPNet_VOC --data_type region_32
# python active_testing.py --model_data_type DeepLab_VOC --data_type region_32
python active_testing.py --model_data_type SEGNet_VOC --data_type region_32
python active_testing.py --model_data_type FCN_VOC --data_type region_32

python active_testing.py --model_data_type PSPNet_CITY --data_type region_32
python active_testing.py --model_data_type UNet_CITY --data_type region_32
# python active_testing.py --model_data_type DeepLab_CITY --data_type region_32
python active_testing.py --model_data_type SEGNet_CITY --data_type region_32
python active_testing.py --model_data_type FCN_CITY --data_type region_32

python active_testing.py --model_data_type UNet_VOC --data_type region_16
python active_testing.py --model_data_type PSPNet_VOC --data_type region_16
# python active_testing.py --model_data_type DeepLab_VOC --data_type region_16
python active_testing.py --model_data_type FCN_VOC --data_type region_16
# python active_testing.py --model_data_type SEGNet_VOC --data_type region_16

python active_testing.py --model_data_type PSPNet_CITY --data_type region_16
python active_testing.py --model_data_type UNet_CITY --data_type region_16
# python active_testing.py --model_data_type SEGNet_CITY --data_type region_16
# python active_testing.py --model_data_type DeepLab_CITY --data_type region_16
python active_testing.py --model_data_type FCN_CITY --data_type region_16