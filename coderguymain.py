import pandas as pd
import sklearn.neural_network
import numpy as np

X = pd.read_csv("train.csv")
y = X["SalePrice"]
X.drop(["SalePrice"], axis=1, inplace=True)

MSZoning = X["MSZoning"].tolist()
currentzones = []

for i in range(len(MSZoning)):
    newZone = True
    if i == 0:
        currentzones.append(MSZoning[i])
    for m in range(len(currentzones)):
        if MSZoning[i] == currentzones[m]:
            newZone = False
    if newZone == True:
        currentzones.append(MSZoning[i])
    
    for m in range(len(currentzones)):
        if MSZoning[i] == currentzones[m]:
            MSZoning[i] = m
          
X["MSZoning"] = MSZoning      