import pandas as pd
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

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

"""
run=MLP.fit_transform(X)"""
MLP=MLPRegressor()

param_dist = {"hidden_layer_sizes": [(100,)],
              "random_state": [None],
              "max_iter": [200]}

random_search = RandomizedSearchCV(MLP, param_distributions=param_dist,
                                   n_iter=20, cv=5)
random_search.fit(X, y)

testdf=pd.read_csv("test.csv",index_col=0)
Xtest=testdf
testcols=Xtest.columns.tolist()

Xtest=cleanup(Xtest,testcols)
print(Xtest.head)


prediction=pd.DataFrame(random_search.predict(Xtest), columns = ["SalePrice"])

prediction.index.name = "Id"

prediction.index+=1461

prediction.to_csv("housePrediction2.csv")