# -*- coding: utf-8 -*-
"""
Created on Mon May 25 23:52:25 2020

@author: cmj
"""
from preproc.count_meta import count_imfq_samemeta
dict_im_fq = count_imfq_samemeta('../data/ISIC20/train.csv',meta_format = 20)
#Counter({'anterior torso': 6915,
#         'upper extremity': 2910,
#         'posterior torso': 2787,
#         'lower extremity': 4990,
#         nan: 2631,
#         'lateral torso': 54,
#         'head/neck': 4587,
#         'palms/soles': 398,
#         'oral/genital': 59})
#
#
#
#
#2090 9636 1412 616 1613 156 173
#4884 13695 3385 1503 2728 280 287