import pandas as pd
from sklearn.neural_network import MLPRegressor
import numpy as np
import random
from sklearn.model_selection import train_test_split

MLP=MLPRegressor(hidden_layer_sizes=(random.randint(50, 500),), activation=random.choice(['relu', 'tanh', 'logistic', 'identity']), alpha=random.uniform(0.001, 0.0001), batch_size='auto', learning_rate=random.choice(['constant', 'invscaling', 'adaptive']), learning_rate_init=random.uniform(0.05, 0.0005), power_t=random.uniform(0.2, 0.8), max_iter=1461, shuffle=random.choice([True, False]), random_state=None, tol=random.uniform(0.001, 0.00001), verbose=random.choice([True, False]), warm_start=random.choice([True, False]), momentum=random.uniform(0.1, 1), nesterovs_momentum=random.choice([True, False]), early_stopping=random.choice([True, False]), validation_fraction=random.uniform(0.01, 1), beta_1=random.uniform(0.1, 1), beta_2=random.uniform(0.001, 0.999), epsilon=random.uniform(10**-7, 10**-9), n_iter_no_change=random.randint(5, 20), max_fun=random.randint(1500, 150000))
times = 0

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

prediction.index+=1461

prediction.to_csv("housePrediction.csv")