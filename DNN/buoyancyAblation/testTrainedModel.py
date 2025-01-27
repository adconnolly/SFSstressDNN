import numpy as np
import xarray as xr
import pickle
import torch
#from torch.autograd import Variable
import torch.utils.data as Data
from e2cnn import nn
from e2cnn import gspaces
#from e2cnn import group
#from torch import nn, optim
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import r2_score
from util import *

size=3
scaleStr='local'
saveFile='C4_bInc_coarseGridReExtrap_'+scaleStr+'_'
bInc=True
Nruns=5
files=["coarse4x2052_Re900.nc","coarse4x40104_Re1800.nc"]
testFiles=["coarse4x60156_Re2700.nc"]

fileUgs=[0.025,0.075]
fileRes=[900.,2700.]
fileB0s=[0.0005, 0.0349]
filemaskpercents=[1./10,1]
valMaskPercent = 0.5

testfileUgs=[0.05]
testfileRes=[1800.]
testfileB0s=[0.0044]
testfilemaskpercents=[0.01] 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

save=1
savePath='./trainedModels/'
for n in range(len(files)):
    isep = files[n].index('_')
    saveFile=saveFile+files[n][6:isep]+files[n][isep+1:-3]+'_'
print(saveFile)
loadDict=saveFile[:-1]+'.pkl'
try:
    auxDataDict=pickle.load( open( savePath+loadDict, "rb" ) )
    noDict=False
except:
    noDict=True

Nhid = [512,256,128,64] # Multiplicity of features in hidden layers associated with the regular representation 
y_text = ["tau_11", "tau_12", "tau_13","tau_22", "tau_23", "tau_33"]
r=np.empty((Nruns,len(y_text))) 
r2=np.empty((Nruns,len(y_text)))
for irun in range(Nruns):

    loadFile=saveFile+str(irun)+'.pt'

    print("Test Files:")
    if bInc:
        if noDict:
            xtest, ytest, tauScaletest, testMask = preprocess(testFiles, testfilemaskpercents, scaleStr, testfileUgs, testfileRes, testfileB0s, size, irun)
            mask =  np.random.rand(xtest.shape[0]) > valMaskPercent
        else:
            xtest, ytest, tauScaletest, testMask = preprocess_dataAug(testFiles, testfilemaskpercents, scaleStr, testfileUgs, testfileRes, testfileB0s, size, irun, krotAll=[0],maskdict=auxDataDict)
            mask =  auxDataDict["valMask_"+str(irun)]
    else:
        if noDict:
            xtest, ytest, tauScaletest, testMask = preprocess_noB(testFiles, testfilemaskpercents, scaleStr, testfileUgs, testfileRes, testfileB0s, size, irun)
            mask =  np.random.rand(xtest.shape[0]) > valMaskPercent
        else:
            xtest, ytest, tauScaletest, testMask = preprocess_dataAug_noB(testFiles, testfilemaskpercents, scaleStr, testfileUgs, testfileRes, testfileB0s, size, irun, krotAll=[0],maskdict=auxDataDict)
            mask =  auxDataDict["valMask_"+str(irun)]

    x_test=xtest[mask]
    del xtest
    y_test=ytest[mask]
    del ytest
    tauScale_test=tauScaletest[mask]
    del tauScaletest
    gc.collect()

    model=CNDNN(Nhid,N=4,size=size,device=device).float().to(device)
    model.load_state_dict(torch.load(savePath+saveFile+str(irun)+'.pt',map_location=device),strict=False)
    model.eval()

    ypred = model(torch.from_numpy(x_test).float().to(device)).cpu().detach().numpy().squeeze()
    y_pred=ypred*tauScale_test
    for i in range(y_pred.shape[1]):
        r2[irun,i]=r2_score(y_test[:,i], y_pred[:,i])
        r[irun,i]=np.corrcoef(y_test[:,i], y_pred[:,i])[0, 1]
#         print("Skills for "+y_text[i])
#         print("R^2: %.4f" % r2[irun,i] )
#         print("Correlation: %.4f" % +r[irun,i]+"\n")


print("Test data")
print(r)
print(r2)

for v in range(r2.shape[1]):
        print(y_text[v]+' avg. R^2 is '+str(np.mean(r2[:,v]))+' +/- '+str(np.std(r2[:,v])))
print('Overall avg. R^2 is '+str(np.mean(r2))+' +/- '+str(np.std(np.mean(r2,axis=1))))
