preproc steps:
1. get all extra images, generate out_csv 
    preproc_extra_UNK1.py  
2. get all training images using color_adj 
    preproc_coloradj_all.py
3. get all testing images using color_adj, also sometimes used as unsupervised data for training
    preproc_coloradj_all_usp.py(19)  preproc_coloradj_test.py(18)
4. generate table (ISIC18/19) for training:
    preproc_all18_info.py
5. generate table (ISIC18/19) for testng:
    preproc_all18_info_usp.py(19)  preproc_all18_info_usp.py(18)
    


# a new extra data extraction
1. preproc_extra_UNK2.py  
2. coloradj: preproc_coloradj_allextra.py
3. generate all18info only for ISIC18: convert_traincsv_18_to_20_noextra.py
4. generate extrainfo: preproc_all20_info_extra.py  (18,19)
   
   
5. a new yaml with three set