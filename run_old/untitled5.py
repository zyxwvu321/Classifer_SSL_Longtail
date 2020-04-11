#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:08:35 2020

@author: minjie
"""
%run  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net resnet50_singleview --out_dir ../checkpoints/resnet50_singleview_fast  --num_epochs 50 --lr 0.005
%run  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net resnet50_twoview --out_dir ../checkpoints/resnet50_twoview_fast  --num_epochs 50 --lr 0.005
%run  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net resnet50_meta --out_dir ../checkpoints/resnet50_meta_fast  --num_epochs 50 --lr 0.005





%run  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net resnet50_singleview --out_dir ../checkpoints/resnet50_singleview_fast_k5  --num_epochs 50 --lr 0.005 --K_fold 5
%run  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net resnet50_twoview --out_dir ../checkpoints/resnet50_twoview_fast_k5  --num_epochs 50 --lr 0.005 --K_fold 5



%run  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net resnet50_metatwoview --out_dir ../checkpoints/resnet50_metatwoview_fast_k5  --num_epochs 30 --lr 0.005 --K_fold 5 --num_workers 8




%run  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net resnet50_singleview --out_dir ../checkpoints/resnet50_singleview_fast_k5  --num_epochs 30 --lr 0.005 --K_fold 5 --num_workers 8
%run  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net resnet50_metasingleview --out_dir ../checkpoints/resnet50_metasingleview_fast_k5  --num_epochs 30 --lr 0.005 --K_fold 5 --num_workers 8



%run  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_singleview --out_dir ../checkpoints/effnetb4_singleview_fast_k5  --num_epochs 30 --lr 0.005 --K_fold 5 --num_workers 8
%run  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_metasingleview --out_dir ../checkpoints/effnetb4_metasingleview_fast_k5  --num_epochs 30 --lr 0.005 --K_fold 5 --num_workers 8




