# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 23:37:36 2019

@author: cmj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('D:\dataset\ISIC\ISIC07_task1_train\info.csv')
datas = df.values

xx = datas[:,2]
yy = datas[:,3]


ww = datas[:,6]
hh = datas[:,7]

rt_w = 256.0/xx
rt_h = 256.0/yy


k =np.minimum(rt_w,rt_h)


ww_nor = ww*k/256.0
hh_nor = hh*k/256.0

plt.scatter(ww_nor,hh_nor)