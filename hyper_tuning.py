import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

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

alpha_values = []
loss_values = []
cross_val_scores = []

for iter_alpha in range(50, 300, 50):
    alpha_values.append(iter_alpha/100)
    model = MLPRegressor(alpha=iter_alpha/100, max_iter=1000)
    print('fitting...')
    model.fit(X,y)
    #low loss + low valid score = overfit
    bestloss = model.best_loss_
    loss_values.append(bestloss)
    print('validating...')
    cross_val_scores.append(cross_val_score(model, X, y, cv=5))

for x in range(len(alpha_values)):
    print(f"alpha: {alpha_values[x]}, loss: {loss_values[x]} validation_avg: {cross_val_scores[x].mean()}, val_std: {cross_val_scores[x].std()}")