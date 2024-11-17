import pandas as pd
import sklearn.neural_network as MLPClassifier
import numpy as np

traindf = pd.read_csv("train.csv").drop(["Id"], inplace=False, axis=1)

#cleanup
traindf.drop(["MSZoning", "Alley", "Neighborhood"], axis=1)

#df.loc[x, col_name] safer access
for x in range(len(traindf["Street"])):
    index = traindf["Street"]
    if index == "Pave":
        traindf.loc[x, "Street"] = 0
    elif index == "Grvl":
        traindf.loc[x, "Street"] = 1

for x in range(len(traindf["HouseStyle"])):
    index = traindf["HouseStyle"][x]
    if index == "1Story":
        traindf.loc[x, "HouseStyle"] = 0
    elif index == "2Story":
        traindf.loc[x, "HouseStyle"] = 1

for x in range(len(traindf["LotShape"])):
    index = traindf["LotShape"][x]
    if index == "Reg":
        traindf["LotShape"][x] = 0
    if index == "IR1":
        traindf["LotShape"][x] = 1
    if index == "IR2": 
        traindf["LotShape"][x] = 2


X = traindf.drop(["SalePrice"], axis=1)
y = traindf["SalePrice"]

print(traindf["HouseStyle"])