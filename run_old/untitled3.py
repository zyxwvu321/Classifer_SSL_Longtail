#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:32:39 2019

@author: minjie
"""

%run train_ISIC.py --imbalance_batchsampler 1 --datasets ./data/ISIC/train_18 --validation_dataset ./data/ISIC/valid_18 --log_file res50.log
%run train_ISIC.py --imbalance_batchsampler 0 --datasets ./data/ISIC/train_18 --validation_dataset ./data/ISIC/valid_18 --log_file res50i.log

%run train_ISIC.py --imbalance_batchsampler 1 --net  resnet34 --datasets ./data/ISIC/train_18 --validation_dataset ./data/ISIC/valid_18 --log_file res34.log
%run train_ISIC.py --imbalance_batchsampler 0 --net  resnet34 --datasets ./data/ISIC/train_18 --validation_dataset ./data/ISIC/valid_18 --log_file res34i.log

%run train_ISIC.py --imbalance_batchsampler 1 --net  resnet101 --datasets ./data/ISIC/train_18 --validation_dataset ./data/ISIC/valid_18 --log_file res101.log
%run train_ISIC.py --imbalance_batchsampler 0 --net  resnet101 --datasets ./data/ISIC/train_18 --validation_dataset ./data/ISIC/valid_18 --log_file res101i.log



%run train_ISIC.py --imbalance_batchsampler 1 --net  resnet50
%run train_ISIC.py --imbalance_batchsampler 1 --net  resnext50_32x4d
%run train_ISIC.py --imbalance_batchsampler 1 --net  se_resnet50
%run train_ISIC.py --imbalance_batchsampler 1 --net  se_resnext50_32x4d

%run train_ISIC.py --imbalance_batchsampler 1 --net  resnet101
%run train_ISIC.py --imbalance_batchsampler 1 --net  resnext101_32x8d




%run predict_ISIC.py --batch_size 1  --net resnet101 --model_file ./sav_model/resnet101-pct80-08.pth --is_valid 0 --datasets /home/minjie/dataset/ISIC/ISIC2018_Task3_Test_Input --fn_csv ./submit/resnet101_08.csv
%run predict_ISIC.py --batch_size 1  --net resnet50 --model_file ./sav_model/resnet50-pct80-08.pth --is_valid 0 --datasets /home/minjie/dataset/ISIC/ISIC2018_Task3_Test_Input --fn_csv ./submit/resnet50_08.csv
%run predict_ISIC.py --batch_size 1  --net resnet50 --model_file ./sav_model/resnet50-pct80-0809.pth --is_valid 0 --datasets /home/minjie/dataset/ISIC/ISIC2018_Task3_Test_Input --fn_csv ./submit/resnet50_0809.csv



%run predict_ISIC.py --batch_size 1  --net resnet50 --model_file ./sav_model/resnet50-pct80-08.pth --is_valid 0 --datasets /home/minjie/dataset/ISIC/ISIC2018_Task3_Test_Input --fn_csv ./submit/resnet50_08_tta.csv --is_tta True --n_tta 10



%run predict_ISIC.py --batch_size 1 --net resnet50 --model_file ./sav_model/resnet50-pct80-08.pth --datasets ./data/ISIC/valid_18 --is_tta True --n_tta 10 --is_valid 1


%run predict_ISIC.py --batch_size 1 --net efficientnet_b2 --model_file ./sav_model/efficientnet_b2-pct80-18-halfepoch.pth --is_valid 0 --datasets /home/minjie/dataset/ISIC/ISIC2018_Task3_Test_Input --fn_csv ./submit/efficientnet_b2_18.csv
%run predict_ISIC.py --batch_size 1 --net efficientnet_b2 --model_file ./sav_model/efficientnet_b2-pct80-18-halfepoch.pth --is_valid 1 --datasets ./data/ISIC/valid_18





%run predict_ISIC.py --net resnet50 --model_file ./sav_model/resnet50-im-ce-18.pth --is_valid 0 --datasets /home/minjie/dataset/ISIC/ISIC2018_Task3_Test_Input --fn_csv ./submit/resnet50-im-ce-18.csv
%run predict_ISIC.py --net resnet50 --model_file ./sav_model/resnet50-im-ce-18.pth --is_valid 1 --datasets ./data/ISIC/all18


%run predict_ISIC.py --net resnet50 --model_file ./sav_model/resnet50-lw-ce-18.pth --is_valid 0 --datasets /home/minjie/dataset/ISIC/ISIC2018_Task3_Test_Input --fn_csv ./submit/resnet50-lw-ce-18.csv
%run predict_ISIC.py --net resnet50 --model_file ./sav_model/resnet50-lw-ce-18.pth --is_valid 1 --datasets ./data/ISIC/all18



%run predict_ISIC.py --net resnet50 --model_file ./sav_model/resnet50-im-bce-18.pth --is_valid 0 --datasets /home/minjie/dataset/ISIC/ISIC2018_Task3_Test_Input --fn_csv ./submit/resnet50-im-bce-18.csv
%run predict_ISIC.py --net resnet50 --model_file ./sav_model/resnet50-im-bce-18.pth --is_valid 1 --datasets ./data/ISIC/all18



%run predict_ISIC.py --net resnet50 --model_file ./sav_model/resnet50-im-ces-18.pth --is_valid 0 --datasets /home/minjie/dataset/ISIC/ISIC2018_Task3_Test_Input --fn_csv ./submit/resnet50-im-ces-18.csv
%run predict_ISIC.py --net resnet50 --model_file ./sav_model/resnet50-im-ces-18.pth  --is_valid 1 --datasets ./data/ISIC/all18



%run predict_ISIC.py --net resnet50 --model_file ./sav_model/resnet50-im-flbce-18.pth --is_valid 0 --datasets /home/minjie/dataset/ISIC/ISIC2018_Task3_Test_Input --fn_csv ./submit/resnet50-im-flbce-18.csv
%run predict_ISIC.py --net resnet50 --model_file ./sav_model/resnet50-im-flbce-18.pth --is_valid 1 --datasets ./data/ISIC/all18


%run predict_ISIC.py --net resnet50 --model_file ./sav_model/resnet50-lw-ce-1819.pth --is_valid 0 --datasets /home/minjie/dataset/ISIC/ISIC2018_Task3_Test_Input --fn_csv ./submit/resnet50-lw-ce-1819.csv
%run predict_ISIC.py --net resnet50 --model_file ./sav_model/resnet50-lw-ce-1819.pth  --is_valid 1 --datasets ./data/ISIC/all18