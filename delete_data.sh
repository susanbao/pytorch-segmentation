data_type=$1

rm -rf ./pro_data/$data_type/train/feature
rm -rf ./pro_data/$data_type/train/entropy
rm -rf ./pro_data/$data_type/train/loss
rm -rf ./pro_data/$data_type/train/target
rm -rf ./pro_data/$data_type/train/image
rm -rf ./pro_data/$data_type/train/output

rm -rf ./pro_data/$data_type/val/feature
rm -rf ./pro_data/$data_type/val/entropy
rm -rf ./pro_data/$data_type/val/loss
rm -rf ./pro_data/$data_type/val/target
rm -rf ./pro_data/$data_type/val/image
rm -rf ./pro_data/$data_type/val/output