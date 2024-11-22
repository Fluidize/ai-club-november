import pandas as pd
import sklearn
import numpy as np

traindf=pd.read_csv("train.csv",index_col=0)


X=traindf.drop(["SalePrice"],inplace=False,axis=1)
y=traindf["SalePrice"]



def cleanup(X):
    for i in range(len(X)):
        data=X[i]
        
        if (data[0]!=int):
            sets=set(data)
            lib={}
            count=0

            for i in set():
                lib.update({i:count})
                count+=1
            
            for i in range(len(data)):
                data[i]=lib[data[i]]
            X[i]=data

    return X

x=cleanup(X)