import pandas as pd
import sklearn
import numpy as np

traindf = pd.read_csv("train.csv")

X = None
y = None

model = None

Xs= traindf.drop(["SalePrice","LotFrontage"],inplace = False, axis = 1)

#---MSZoning
unmszoning = Xs["MSZoning"].tolist()
mszoneunique = Xs["MSZoning"].unique().tolist()
MSZoning = []

for z in range(len(unmszoning)):
    MSZoning.append(mszoneunique.index(unmszoning[z]))

Xs["MSZoning"] = MSZoning

#---Street
unstreet = Xs["Street"].tolist()
streetunique = Xs["Street"].unique().tolist()
Street = []

for z in range(len(unstreet)):
    Street.append(streetunique.index(unstreet[z]))

Xs["Street"] = Street

#---Alley
unalley = Xs["Alley"].tolist()
alleyunique = Xs["Alley"].unique().tolist()
Alley = []

for z in range(len(unalley)):
    Alley.append(alleyunique.index(unalley[z]))

Xs["Alley"] = Alley

#---LotShapelotshape
unlotshape = Xs["LotShape"].tolist()
lotshapeunique = Xs["LotShape"].unique().tolist()
LotShape = []

for z in range(len(unlotshape)):
    LotShape.append(lotshapeunique.index(unlotshape[z]))

Xs["LotShape"] = LotShape

#---LandContour
unlandcontour = Xs["LandContour"].tolist()
landcontourunique = Xs["LandContour"].unique().tolist()
LandContour = []

for z in range(len(unlandcontour)):
    LandContour.append(landcontourunique.index(unlandcontour[z]))

Xs["LandContour"] = LandContour

#---LotConfig
unlotconfig = Xs["LotConfig"].tolist()
lotconfigunique = Xs["LotConfig"].unique().tolist()
LotConfig = []

for z in range(len(unlotconfig)):
    LotConfig.append(lotconfigunique.index(unlotconfig[z]))

Xs["LotConfig"] = LotConfig

#---LandSlope
unlandslope = Xs["LandSlope"].tolist()
landslopeunique = Xs["LandSlope"].unique().tolist()
LandSlope = []

for z in range(len(unlandslope)):
    LandSlope.append(landslopeunique.in   dex(unlandslope[z]))

Xs["LandSlope"] = LandSlope

#---Neighborhood
unneighborhood = Xs["Neighborhood"].tolist()
neighborhoodunique = Xs["Neighborhood"].unique().tolist()
Neighborhood = []

for z in range(len(unneighborhood)):
    Neighborhood.append(neighborhoodunique.index(unneighborhood[z]))

Xs["Neighborhood"] = Neighborhood

#---   