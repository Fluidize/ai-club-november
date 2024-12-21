import pandas as pd
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler

MLP=MLPRegressor(learning_rate_init=0.01, max_iter=999999, hidden_layer_sizes=np.shape(250,))
times = 1

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

for i in range(times):
    X = pd.concat([X, X], ignore_index=True)
    y = pd.concat([y, y], ignore_index=True)

"""
run=MLP.fit_transform(X)"""

MLP.fit(X,y)

testdf=pd.read_csv("test.csv",index_col=0)
Xtest=testdf
testcols=Xtest.columns.tolist()

Xtest=cleanup(Xtest,testcols)
print(Xtest.head)


prediction=pd.DataFrame(MLP.predict(Xtest), columns = ["SalePrice"])

prediction.index.name = "Id"

prediction.index+=1461 * 2**times

prediction.to_csv("housePrediction.csv")