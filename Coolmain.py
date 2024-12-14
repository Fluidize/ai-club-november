import pandas as pd
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
MLP=MLPRegressor()
scaler=StandardScaler()

traindf=pd.read_csv("train.csv",index_col=0)


X=traindf.drop(["SalePrice"],inplace=False,axis=1)
y=traindf["SalePrice"]
cols=X.columns.tolist()
print(cols)
tranfers=[]
setter=[]

# inplement a counter for when I add a list of dictorionary
# add a check for the dictorionary list 
# so that if there is no dictorionary the function makes one
class allcleanup():
    def __init__(self,X,column):
        self=self
        self.X=X
        self.column=column
    def cleanup(self):
        for i in range(len(self.column)):
            self.data=X[self.column[i]].tolist()
        
            if (type(self.data[0])!=int):
                self.sets=set(self.data)
                self.dictor={}
                self.isin=False
                if (i+1)<len(setter):
                        isin=False
                        for l in self.sets:
                            if l in setter[i]:
                                isin=True
                            if isin==False:
                                self.polish(i)
                                break
                        for k in range(len(self.data)):
                            self.data[k]=tranfers[i][self.data[k]]
                        X[self.column[i]]=self.data
                        tranfers.append(self.dictor)
                        setter.append(self.sets)
                    

                else:
                    self.polish(i)
        return X
    def polish(self,section):
        count=0
        for j in self.sets:
            self.dictor.update({j:count})
            count+=1
        for k in range(len(self.data)):
            self.data[k]=self.dictor[self.data[k]]
        X[self.column[section]]=self.data
        tranfers.append(self.dictor)
        setter.append(self.sets)

X=allcleanup(X,cols).cleanup()
print(X.head)

MLP.fit(X,y)




testdf=pd.read_csv("test.csv",index_col=0)
Xtest=testdf
testcols=Xtest.columns.tolist()

Xtest=allcleanup(Xtest,testcols).cleanup()
print(Xtest.head)


predictorion=pd.DataFrame(MLP.predict(Xtest), columns = ["SalePrice"])

predictorion.index.name = "Id"

predictorion.index+=1461

predictorion.to_csv("housePredictorion.csv")