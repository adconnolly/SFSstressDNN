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
scaleStr='stats'
saveFile='C4_midGridReInterp_'+scaleStr+'_'
files=["coarse4x1026_Re900.nc","coarse4x3078_Re2700.nc"]
fileUgs=[0.025,0.075]
fileRes=[900.,2700.]
fileB0s=[0.0005, 0.0349]
filemaskpercents=[1./10,1]
valMaskPercent = 0.5

testFiles=["coarse4x2052_Re1800.nc"]
testfileUgs=[0.05]
testfileRes=[1800.]
testfileB0s=[0.0044]
testfilemaskpercents=[0.4] 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

save=1
savePath='./trainedModels/'
for n in range(len(files)):
    isep = files[n].index('_')
    saveFile=saveFile+files[n][6:isep]+files[n][isep+1:-3]+'_'
print(saveFile)
loadDict=saveFile[:-1]+'.pkl'
auxDataDict=pickle.load( open( savePath+loadDict, "rb" ) )

Nhid = [512,256,128,64] # Multiplicity of features in hidden layers associated with the regular representation 
Nruns=5
y_text = ["tau_11", "tau_12", "tau_13","tau_22", "tau_23", "tau_33"]
r=np.empty((Nruns,len(y_text))) 
r2=np.empty((Nruns,len(y_text)))
for irun in range(Nruns):

    xtrain, y_train, tauScale_train, trainMask = preprocess_dataAug(files, filemaskpercents, 'unscaled', fileUgs, fileRes, fileB0s, size, irun, krotAll = [0], maskdict=auxDataDict, reshape=False )
    
    uv_trainStd=np.std(np.sqrt(xtrain[:,0]**2+xtrain[:,1]**2))
    w_trainStd=np.std(xtrain[:,2])
    b_trainStd=np.std(xtrain[:,3])
    xtrain[:,0]=xtrain[:,0]/uv_trainStd
    xtrain[:,1]=xtrain[:,1]/uv_trainStd
    xtrain[:,2]=xtrain[:,2]/w_trainStd
    xtrain[:,3]=xtrain[:,3]/b_trainStd
    x_train=myreshape(xtrain)
    del xtrain

    ti3_trainStd=np.std(np.sqrt(y_train[:,2]**2
                    +y_train[:,4]**2))
    t33_trainStd=np.std(y_train[:,5])
    th_trainStd=np.std(np.sqrt(y_train[:,0]**2
                  +y_train[:,1]**2
                  +y_train[:,3]**2))

    tauScale_train[:,0]=th_trainStd*tauScale_train[:,0]
    tauScale_train[:,1]=th_trainStd*tauScale_train[:,1]
    tauScale_train[:,2]=ti3_trainStd*tauScale_train[:,2]
    tauScale_train[:,3]=th_trainStd*tauScale_train[:,3]
    tauScale_train[:,4]=ti3_trainStd*tauScale_train[:,4]
    tauScale_train[:,5]=t33_trainStd*tauScale_train[:,5]
    
    print("Test Files:")
    xtest, ytest, tauScaletest, testMask = preprocess_dataAug(testFiles, testfilemaskpercents, 'unscaled', testfileUgs, testfileRes, testfileB0s, size, irun, krotAll=[3], maskdict=auxDataDict, reshape=False)

    xtest[:,0]=xtest[:,0]/uv_trainStd
    xtest[:,1]=xtest[:,1]/uv_trainStd
    xtest[:,2]=xtest[:,2]/w_trainStd
    xtest[:,3]=xtest[:,3]/b_trainStd
    tauScaletest[:,0]=th_trainStd*tauScaletest[:,0]
    tauScaletest[:,1]=th_trainStd*tauScaletest[:,1]
    tauScaletest[:,2]=ti3_trainStd*tauScaletest[:,2]
    tauScaletest[:,3]=th_trainStd*tauScaletest[:,3]
    tauScaletest[:,4]=ti3_trainStd*tauScaletest[:,4]
    tauScaletest[:,5]=t33_trainStd*tauScaletest[:,5]
    
    mask =  auxDataDict["valMask_"+str(irun)]
    x_test=myreshape(xtest[mask])
    del xtest
    y_test=ytest[mask]
    del ytest
    tauScale_test=tauScaletest[mask]
    del tauScaletest
    gc.collect()

    loadFile=saveFile+str(irun)+'.pt'
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
