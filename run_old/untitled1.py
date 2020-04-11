# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 22:23:10 2019

@author: cmj
"""

import logging

logger = logging.getLogger("nick")
logger.setLevel(logging.DEBUG)  

fh = logging.FileHandler("test.log",encoding="utf-8")
ch = logging.StreamHandler()

formatter = logging.Formatter(fmt="%(asctime)s %(name)s %(filename)s %(message)s",datefmt="%Y/%m/%d %X")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

logger.warning("泰拳警告")
logger.info("提示")
logger.error("错误")

import pandas as pd
import numpy as np
aa =np.random.rand(3,3)
cols = ['0','1','2']
df = pd.DataFrame(data=aa,columns = cols)
logger.info('*'*30)
logger.info(df)