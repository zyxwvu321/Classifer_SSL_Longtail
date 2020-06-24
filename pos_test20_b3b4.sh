

python train_BNN.py --config_file ./configs/effb3_meta_default20_extrameta_w_d_metafix.yaml

python test_gen_classifer.py --config_file ./configs/effb3_meta_default20_extrameta_w_d_metafix_test.yaml  MISC.ONLY_TEST True MISC.TTA True MISC.N_TTA 20

python train_BNN.py --config_file ./configs/effb3_meta_default20_extrameta_w_d_metafix_pos.yaml

python test_gen_classifer.py --config_file ./configs/effb3_meta_default20_extrameta_w_d_metafix_pos_test.yaml  MISC.ONLY_TEST True MISC.TTA True MISC.N_TTA 20



python train_BNN.py --config_file ./configs/effb4_meta_default20_extrameta_w_d_metafix.yaml

python test_gen_classifer.py --config_file ./configs/effb4_meta_default20_extrameta_w_d_metafix_test.yaml  MISC.ONLY_TEST True MISC.TTA True MISC.N_TTA 20


python train_BNN.py --config_file ./configs/effb4_meta_default20_extrameta_w_d_metafix_pos.yaml

python test_gen_classifer.py --config_file ./configs/effb4_meta_default20_extrameta_w_d_metafix_pos_test.yaml  MISC.ONLY_TEST True MISC.TTA True MISC.N_TTA 20

