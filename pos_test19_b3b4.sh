#b4
python train_BNN.py --config_file ./configs/effb4_meta_default19_extrameta_w_d.yaml

python test_gen_classifer.py --config_file ./configs/effb4_meta_default19_extrameta_w_d_test.yaml  MISC.ONLY_TEST True MISC.TTA True MISC.N_TTA 20

python train_BNN.py --config_file ./configs/effb4_meta_default19_extrameta_w_d_pos.yaml

python test_gen_classifer.py --config_file ./configs/effb4_meta_default19_extrameta_w_d_pos_test.yaml  MISC.ONLY_TEST True MISC.TTA True MISC.N_TTA 20


python train_BNN.py --config_file ./configs/effb4_meta_default19_extrameta_w_fl_d.yaml

python test_gen_classifer.py --config_file ./configs/effb4_meta_default19_extrameta_w_fl_d_test.yaml  MISC.ONLY_TEST True MISC.TTA True MISC.N_TTA 20



#b3



python train_BNN.py --config_file ./configs/effb3_meta_default19_extrameta_w_d.yaml

python test_gen_classifer.py --config_file ./configs/effb3_meta_default19_extrameta_w_d_test.yaml  MISC.ONLY_TEST True MISC.TTA True MISC.N_TTA 20

python train_BNN.py --config_file ./configs/effb3_meta_default19_extrameta_w_d_pos.yaml

python test_gen_classifer.py --config_file ./configs/effb3_meta_default19_extrameta_w_d_pos_test.yaml  MISC.ONLY_TEST True MISC.TTA True MISC.N_TTA 20


python train_BNN.py --config_file ./configs/effb3_meta_default19_extrameta_w_fl_d.yaml

python test_gen_classifer.py --config_file ./configs/effb3_meta_default19_extrameta_w_fl_d_test.yaml  MISC.ONLY_TEST True MISC.TTA True MISC.N_TTA 20
