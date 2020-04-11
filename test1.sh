#python test_gen_classifer.py --config_file ./configs/resnet50_meta_pcs.yaml MISC.TTA False MISC.N_TTA 1
#python test_gen_classifer.py --config_file ./configs/effb4_meta_pcs.yaml MISC.TTA False MISC.N_TTA 1


#python test_gen_classifer.py --config_file ./configs/resnet50_meta_pcs.yaml MISC.TTA True MISC.N_TTA 10
#python test_gen_classifer.py --config_file ./configs/effb4_meta_pcs.yaml MISC.TTA True MISC.N_TTA 10




python test_gen_classifer.py --config_file ./configs/resnet50_meta_pcs.yaml    DATASETS.ROOT_DIR  ../data/ISIC18/task3/ISIC2018_Task3_Test_Input_coloradj  DATASETS.INFO_CSV ./dat/ISIC18_info_test.csv  MISC.TTA False MISC.N_TTA 1 MISC.ONLY_TEST True  MISC.LOGFILE_EVAL log_eval_test.txt  
python test_gen_classifer.py --config_file ./configs/resnet50_meta_pcs.yaml    DATASETS.ROOT_DIR  ../data/ISIC18/task3/ISIC2018_Task3_Test_Input_coloradj  DATASETS.INFO_CSV ./dat/ISIC18_info_test.csv  MISC.TTA True MISC.N_TTA 10 MISC.ONLY_TEST True MISC.LOGFILE_EVAL log_eval_test.txt

python test_gen_classifer.py --config_file ./configs/effb4_meta_pcs.yaml    DATASETS.ROOT_DIR  ../data/ISIC18/task3/ISIC2018_Task3_Test_Input_coloradj  DATASETS.INFO_CSV ./dat/ISIC18_info_test.csv  MISC.TTA False MISC.N_TTA 1 MISC.ONLY_TEST True MISC.LOGFILE_EVAL log_eval_test.txt
python test_gen_classifer.py --config_file ./configs/effb4_meta_pcs.yaml    DATASETS.ROOT_DIR  ../data/ISIC18/task3/ISIC2018_Task3_Test_Input_coloradj  DATASETS.INFO_CSV ./dat/ISIC18_info_test.csv  MISC.TTA True MISC.N_TTA 10 MISC.ONLY_TEST True MISC.LOGFILE_EVAL log_eval_test.txt

