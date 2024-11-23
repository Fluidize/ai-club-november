import pandas as pd
import sklearn
import numpy as np

traindf=pd.read_csv("train.csv",index_col=0)


X=traindf.drop(["SalePrice"],inplace=False,axis=1)
y=traindf["SalePrice"]
cols=X.columns.tolist()
print(cols)

def cleanup(X,column):
    for i in range(len(column)):
        data=X[column[i]].tolist()
    
        if (type(data[0])!=int):
            sets=set(data)
            lib={}
            count=0


            for j in sets:
                lib.update({j:count})
                count+=1
            



            for k in range(len(data)):
                data[k]=lib[data[k]]
            X[column[i]]=data

    return X

X=cleanup(X,cols)
print(X.head)


