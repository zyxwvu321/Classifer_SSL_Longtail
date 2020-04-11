
python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_singleview --out_dir ../checkpoints/effnetb4_singleview_fast_k1_ce1  --num_epochs 60 --lr 0.001 --K_fold 1 --num_workers 8 --loss_type ce
python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_singleview --out_dir ../checkpoints/effnetb4_singleview_fast_k1_fl1  --num_epochs 60 --lr 0.001 --K_fold 1 --num_workers 8 --loss_type focalloss

python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_singleview --out_dir ../checkpoints/effnetb4_singleview_fast_k1_lbce1  --num_epochs 60 --lr 0.001 --K_fold 1 --num_workers 8 --loss_type ce_smooth





python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_metasingleview --out_dir ../checkpoints/effnetb4_metasingleview_fast_k1_ce1  --num_epochs 60 --lr 0.001 --K_fold 1 --num_workers 8 --loss_type ce
python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_metasingleview --out_dir ../checkpoints/effnetb4_metasingleview_fast_k1_fl1  --num_epochs 60 --lr 0.001 --K_fold 1 --num_workers 8 --loss_type focalloss

python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_metasingleview --out_dir ../checkpoints/effnetb4_metasingleview_fast_k1_lbce1 --num_epochs 60 --lr 0.001 --K_fold 1 --num_workers 8 --loss_type ce_smooth



python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_singleview --out_dir ../checkpoints/effnetb4_singleview_fast_k1_ce1_cbam  --num_epochs 60 --lr 0.001 --K_fold 1 --num_workers 8 --loss_type ce --cbam 1
python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_singleview --out_dir ../checkpoints/effnetb4_singleview_fast_k1_fl1_cbam  --num_epochs 60 --lr 0.001 --K_fold 1 --num_workers 8 --loss_type focalloss --cbam 1

python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_singleview --out_dir ../checkpoints/effnetb4_singleview_fast_k1_lbce1_cbam  --num_epochs 60 --lr 0.001 --K_fold 1 --num_workers 8 --loss_type ce_smooth --cbam 1





python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_metasingleview --out_dir ../checkpoints/effnetb4_metasingleview_fast_k1_ce1_cbam  --num_epochs 60 --lr 0.001 --K_fold 1 --num_workers 8 --loss_type ce --cbam 1
python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_metasingleview --out_dir ../checkpoints/effnetb4_metasingleview_fast_k1_fl1_cbam  --num_epochs 60 --lr 0.001 --K_fold 1 --num_workers 8 --loss_type focalloss --cbam 1

python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_metasingleview --out_dir ../checkpoints/effnetb4_metasingleview_fast_k1_lbce1_cbam --num_epochs 60 --lr 0.001 --K_fold 1 --num_workers 8 --loss_type ce_smooth --cbam 1