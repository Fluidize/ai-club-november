import pandas as pd
import sklearn.neural_network
import numpy as np
import math

def findEach(column):
    instances = []
    for i in range(len(column)):
        newInstance = True
        if i == 0:
            instances.append(column[i])
        for m in range(len(instances)):
            if column[i] == instances[m]:
                newInstance = False
        if newInstance == True:
            instances.append(column[i])
    print(instances)
    return instances

X = pd.read_csv("train.csv")
y = X["SalePrice"]
X.drop(["SalePrice", "Alley"], axis=1, inplace=True)

MSZoning = X["MSZoning"].tolist()

currentzones = findEach(MSZoning)
for i in range(len(MSZoning)):
    for m in range(len(currentzones)):
        if MSZoning[i] == currentzones[m]:
            MSZoning[i] = m
          
X["MSZoning"] = MSZoning      

LotFrontage = X["LotFrontage"].tolist()

for i in range(len(LotFrontage)):
    if math.isnan(LotFrontage[i]) == True:
        LotFrontage[i] = 0
    LotFrontage[i] = int(LotFrontage[i])
X["LotFrontage"] = LotFrontage

Street = X["Street"].tolist()

streets = findEach(Street)
for i in range(len(Street)):
    for m in range(len(streets)):
        if Street[i] == streets[m]:
            Street[i] = m

X["Street"] = Street
