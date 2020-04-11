

python train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net resnet50_singleview --out_dir ../checkpoints/resnet50_singleview_fast_k5  --num_epochs 60 --lr 0.002 --K_fold 5 --num_workers 8 --only_test 1 --tta 0 --log_file eval_log.txt
python train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net resnet50_singleview --out_dir ../checkpoints/resnet50_singleview_fast_k5  --num_epochs 60 --lr 0.002 --K_fold 5 --num_workers 8 --only_test 1 --tta 1 --log_file eval_log.txt


python train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net resnet50_metasingleview --out_dir ../checkpoints/resnet50_metasingleview_fast_k5  --num_epochs 60 --lr 0.002 --K_fold 5 --num_workers 8 --only_test 1 --tta 0 --log_file eval_log.txt
python train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net resnet50_metasingleview --out_dir ../checkpoints/resnet50_metasingleview_fast_k5  --num_epochs 60 --lr 0.002 --K_fold 5 --num_workers 8 --only_test 1 --tta 1 --log_file eval_log.txt



python train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_singleview --out_dir ../checkpoints/effnetb4_singleview_fast_k5  --num_epochs 60 --lr 0.002 --K_fold 5 --num_workers 8 --only_test 1 --tta 0 --log_file eval_log.txt
python train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_singleview --out_dir ../checkpoints/effnetb4_singleview_fast_k5  --num_epochs 60 --lr 0.002 --K_fold 5 --num_workers 8 --only_test 1 --tta 1 --log_file eval_log.txt


python train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_metasingleview --out_dir ../checkpoints/effnetb4_metasingleview_fast_k5  --num_epochs 60 --lr 0.002 --K_fold 5 --num_workers 8 --only_test 1 --tta 0 --log_file eval_log.txt
python train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_metasingleview --out_dir ../checkpoints/effnetb4_metasingleview_fast_k5  --num_epochs 60 --lr 0.002 --K_fold 5 --num_workers 8 --only_test 1 --tta 1 --log_file eval_log.txt






python train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net resnet50_singleview --out_dir ../checkpoints/resnet50_singleview_fast_k5  --num_epochs 60 --lr 0.002 --K_fold 5 --num_workers 8
python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net resnet50_metasingleview --out_dir ../checkpoints/resnet50_metasingleview_fast_k5  --num_epochs 60 --lr 0.002 --K_fold 5 --num_workers 8



python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_singleview --out_dir ../checkpoints/effnetb4_singleview_fast_k5  --num_epochs 60 --lr 0.002 --K_fold 5 --num_workers 8
python  train_ISIC_gllcmeta.py  --datasets ../data/ISIC18/task3/ISIC2018_Task3_Training_Input_coloradj  --net effnetb4_metasingleview --out_dir ../checkpoints/effnetb4_metasingleview_fast_k5  --num_epochs 60 --lr 0.002 --K_fold 5 --num_workers 8

