python train_ISIC.py --imbalance_batchsampler 1 --log_file resnet50.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid
python train_ISIC.py --imbalance_batchsampler 0 --log_file resnet50i.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid

python train_ISIC.py --imbalance_batchsampler 1 --log_file resnet50_bce.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid --loss_type bce
python train_ISIC.py --imbalance_batchsampler 0 --log_file resnet50i_bce.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid --loss_type bce

python train_ISIC.py --imbalance_batchsampler 1 --log_file resnet50_ces.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid --loss_type ce_smooth
python train_ISIC.py --imbalance_batchsampler 0 --log_file resnet50i_ce.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid --loss_type ce_smooth


python train_ISIC.py --imbalance_batchsampler 1 --net  resnet101 --log_file resnet101.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid
python train_ISIC.py --imbalance_batchsampler 1 --net  resnet34  --log_file resnet34.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid



train_ISIC.py --imbalance_batchsampler 1  --net  efficientnet_b2 --log_file efficientnet_b2.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid
train_ISIC.py --imbalance_batchsampler 1  --net  efficientnet_b3 --log_file efficientnet_b3.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid
train_ISIC.py --imbalance_batchsampler 1  --net  efficientnet_b0 --log_file efficientnet_b0.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid




python train_ISIC.py --imbalance_batchsampler 1 --net  resnext50_32x4d  --log_file resnext50_32x4d.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid
python train_ISIC.py --imbalance_batchsampler 1 --net  se_resnet50 --log_file se_resnet50.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid_18
python train_ISIC.py --imbalance_batchsampler 1 --net  se_resnext50_32x4d --log_file se_resnext50_32x4d.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid


python train_ISIC.py --imbalance_batchsampler 1 --log_file resnet50_lbsmooth.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid --loss_type ce_smooth
python train_ISIC.py --imbalance_batchsampler 1 --log_file resnet50_bce.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid --loss_type bce


python train_ISIC.py --imbalance_batchsampler 1 --net  resnext101_32x8d  --log_file resnext101_32x8d.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid
python train_ISIC.py --imbalance_batchsampler 1 --net  se_resnext101_32x4d --log_file se_resnext101_32x4d.log --datasets ./data/ISIC/all --validation_dataset ./data/ISIC/valid






