# -*- coding: utf-8 -*-


dict_y = {0:[0,1,2,3,4],0.5:[0.5,1.0,1.5],1.0:[0,0.5,1,1.5,2,3,4],1.5:[0.5,1.0,1.5,2.0,2.5],2.0:[0,1,1.5,2,2.5,3,4],\
          2.5:[1.5,2.0,2.5,3.0,3.5],3.0:[0,1,2,2.5,3,3.5,4],3.5:[2.5,3.0,3.5],4:[0,1,2,3,4]}
n=0
for key,val in dict_y.items():
    for i1 in range(len(val)-1):
        for i2 in range(i1+1,len(val)):
            v1,v2 = val[i1],val[i2]
            ll = v2-v1
            if (ll+key) in dict_y.keys() and v1 in dict_y[ll+key] and v2 in dict_y[ll+key]:
                n = n + 1
print(n)
                