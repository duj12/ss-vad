#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/data/megastore/Projects/DuJing/code/Datadriven-VAD


stage=$1
data_dir=ZhEn_1w
if [ $stage -eq 0 ] ; then 
#首先准备好wav.list(音频路径列表文件)， wav.label(帧级别VAD标签:0-静音,1-语音)
#wav.list:
# filename
# xxxxx/xxxx/xxx/xxxx.wav
#wav.label:
# label
# 0 0 0 1 1 1 1 1 1 1 0 1 1 1 1 1 0 0

#特征，硬标签和软标签
mkdir -p   data/$data_dir/hdf5_mfcc_clean  data/$data_dir/hardlabels data/$data_dir/softlabels

#提取干净语音的梅尔特征，方便用于生成软标签。
for split in valid   ; do 
# make utt-label pair file 
# cat data/$data_dir/$split.list  | awk -F"/" '{print $NF}'  | paste - data/$data_dir/$split.label > data/$data_dir/hardlabels/$split.label || exit 1;

mkdir -p data/$data_dir/hdf5_mfcc_clean/$split
mkdir -p data/$data_dir/softlabels/$split
#echo "pwd2GPU" | sudo chmod -R 777 ./
feat_csv=/data/megastore/Projects/DuJing/code/Datadriven-VAD/data/$data_dir/hdf5_mfcc_clean/$split.csv

# extract mfcc
python3 -u data/extract_feature.py  data/$data_dir/$split.list  $feat_csv  -o data/$data_dir/hdf5_mfcc_clean/$split  

# extract soft labels
python3 -u data/prepare_labels.py --pre pretrained_models/teacher1/model.pth  $feat_csv  data/$data_dir/softlabels/$split data/$data_dir/softlabels/$split.csv # || exit 1 ;

done

fi

#中间进行了数据加噪，然后提取加噪音频的特征
if [ $stage -eq 1 ] ; then 

#提取干净语音的梅尔特征，方便用于生成软标签。
for split in valid  train ; do 
mkdir -p data/$data_dir/hdf5_mfcc_noisy/$split
noisy_data_dir=/data/megastore/Projects/DuJing/data/VAD_noisy
feat_csv=/data/megastore/Projects/DuJing/code/Datadriven-VAD/data/$data_dir/hdf5_mfcc_noisy/$split.csv

# extract mfcc
python3 -u data/extract_feature.py -c 1 $noisy_data_dir/$split.list  $feat_csv  -o data/$data_dir/hdf5_mfcc_noisy/$split  

done

fi


if [ $stage -le 2 ] ;then
CUDA_VISIBLE_DEVICES=0 \
python3 -u run.py train configs/ZhEn_1w_noisy.yaml

fi