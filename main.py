import pandas as pd
from sklearn.neural_network import MLPRegressor

model=MLPRegressor(max_iter=500)

traindf=pd.read_csv("train.csv",index_col=0)

X=traindf.drop(["SalePrice"],inplace=False,axis=1)
y=traindf["SalePrice"]

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

cols=X.columns.tolist()
X=cleanup(X,cols)

print('TRAINING')
model.fit(X,y)

testdf=pd.read_csv("test.csv",index_col=0)

testcols=testdf.columns.tolist()
testdf=cleanup(testdf,testcols)

prediction=pd.DataFrame(model.predict(testdf), columns = ["SalePrice"])

prediction.index.name = "Id"

prediction.index+=1461

prediction.to_csv("prediction.csv")