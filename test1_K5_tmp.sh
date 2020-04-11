#python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_singleview --out_dir ../checkpoints/effnetb4_singleview_fast_k1_ce1  --num_epochs 60 --lr 0.001 --K_fold 1 --num_workers 8 --loss_type ce
python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net resnet50_singleview --out_dir ../checkpoints/resnet50_singleview_fast_k5_fl1  --num_epochs 60 --lr 0.001 --K_fold 5 --num_workers 8 --loss_type focalloss

#python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_singleview --out_dir ../checkpoints/effnetb4_singleview_fast_k1_lbce2  --num_epochs 60 --lr 0.001 --K_fold 1 --num_workers 8 --loss_type ce_smooth





#python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_metasingleview --out_dir ../checkpoints/effnetb4_metasingleview_fast_k1_ce1  --num_epochs 60 --lr 0.001 --K_fold 1 --num_workers 8 --loss_type ce
python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net resnet50_metasingleview --out_dir ../checkpoints/resnet50_metasingleview_fast_k5_fl1  --num_epochs 60 --lr 0.001 --K_fold 5 --num_workers 8 --loss_type focalloss

#python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_metasingleview --out_dir ../checkpoints/effnetb4_metasingleview_fast_k1_lbce2 --num_epochs 60 --lr 0.001 --K_fold 1 --num_workers 8 --loss_type ce_smooth






python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC19/ISIC_2019_Training_Input_coloradj  --net effnetb4_singleview --out_dir ../checkpoints/effnetb4_singleview_19_k5_lbce2  --num_epochs 60 --lr 0.001 --K_fold 5 --num_workers 8 --loss_type ce_smooth --info_csv ./dat/ISIC19_info.csv --fn_map ./dat/fn_maps_ISIC19.pth

python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC19/ISIC_2019_Training_Input_coloradj  --net effnetb4_metasingleview --out_dir ../checkpoints/effnetb4_metasingleview_19_k5_lbce2  --num_epochs 60 --lr 0.001 --K_fold 5 --num_workers 8 --loss_type ce_smooth --info_csv ./dat/ISIC19_info.csv --fn_map ./dat/fn_maps_ISIC19.pth