
python test_ISIC_gllcmeta.py --net resnet50_singleview --out_dir ../checkpoints/resnet50_singleview_fast_k5 --K_fold 5 --num_workers 8  --tta 0 --log_file eval_log_test.txt
python test_ISIC_gllcmeta.py --net resnet50_singleview --out_dir ../checkpoints/resnet50_singleview_fast_k5 --K_fold 5 --num_workers 8  --tta 1 --log_file eval_log_test.txt
 
 
 
python test_ISIC_gllcmeta.py --net resnet50_metasingleview --out_dir ../checkpoints/resnet50_metasingleview_fast_k5 --K_fold 5 --num_workers 8  --tta 0 --log_file eval_log_test.txt
python test_ISIC_gllcmeta.py --net resnet50_metasingleview --out_dir ../checkpoints/resnet50_metasingleview_fast_k5 --K_fold 5 --num_workers 8  --tta 1 --log_file eval_log_test.txt


python test_ISIC_gllcmeta.py --net effnetb4_singleview --out_dir ../checkpoints/effnetb4_singleview_fast_k5 --K_fold 5 --num_workers 8  --tta 0 --log_file eval_log_test.txt
python test_ISIC_gllcmeta.py --net effnetb4_singleview --out_dir ../checkpoints/effnetb4_singleview_fast_k5 --K_fold 5 --num_workers 8  --tta 1 --log_file eval_log_test.txt

python test_ISIC_gllcmeta.py --net effnetb4_metasingleview --out_dir ../checkpoints/effnetb4_metasingleview_fast_k5 --K_fold 5 --num_workers 8  --tta 0 --log_file eval_log_test.txt
python test_ISIC_gllcmeta.py --net effnetb4_metasingleview --out_dir ../checkpoints/effnetb4_metasingleview_fast_k5 --K_fold 5 --num_workers 8  --tta 1 --log_file eval_log_test.txt


