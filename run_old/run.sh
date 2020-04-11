python train_ISIC.py --imbalance_batchsampler 1 --log_file resnet50.log
python train_ISIC.py --imbalance_batchsampler 1 --net  resnet101 --log_file resnet101.log
python train_ISIC.py --imbalance_batchsampler 1 --net  resnet34  --log_file resnet34.log

python train_ISIC.py --imbalance_batchsampler 1 --net  resnext50_32x4d  --log_file resnext50_32x4d.log
python train_ISIC.py --imbalance_batchsampler 1 --net  se_resnet50 --log_file se_resnet50.log
python train_ISIC.py --imbalance_batchsampler 1 --net  se_resnext50_32x4d --log_file se_resnext50_32x4d.log

python train_ISIC.py --imbalance_batchsampler 1 --net  resnext101_32x8d  --log_file resnext101_32x8d.log
python train_ISIC.py --imbalance_batchsampler 1 --net  se_resnext101_32x4d --log_file se_resnext101_32x4d.log




