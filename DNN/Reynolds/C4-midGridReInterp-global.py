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
sizeStr=str(size)
scaleStr='global'
#saveFile='C4_'+sizeStr+'x'+sizeStr+'_' 
saveFile='C4_midGridReInterp_'+scaleStr+'_' 
files=["coarse4x1026_Re900.nc","coarse4x3078_Re2700.nc"]
fileUgs=[0.025,0.075]
fileRes=[20000.,60000.]
fileB0s=[-0.0005, -0.0349]
filemaskpercents=[1./10,1]
valMaskPercent = 0.5

testFiles=["coarse4x2052_Re1800.nc"]
testfileUgs=[0.05]
testfileRes=[40000.]
testfileB0s=[-0.0044]
testfilemaskpercents=[0.4] 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

save=1
savePath='trainedModels/'
for n in range(len(files)):
    isep = files[n].index('_')
    saveFile=saveFile+files[n][6:isep]+files[n][isep+1:-3]+'_'
print(saveFile)
plotPath='quickPlots/'

Nhid = [512,256,128,64] # Multiplicity of features in hidden layers associated with the regular representation 

Nruns=5
y_text = ["tau_11", "tau_12", "tau_13","tau_22", "tau_23", "tau_33"]
r=np.empty((Nruns,len(y_text))) # 3 is number of outputs, thus the hardcode
r2=np.empty((Nruns,len(y_text)))
auxDataDict=dict()
for irun in range(Nruns):

    print("Train Files:")
    x_train, y_train, tauScale_train, trainMask = preprocess(files, filemaskpercents, scaleStr, fileUgs, fileRes, fileB0s, size, irun)
    auxDataDict.update(trainMask)
    
    BATCH_SIZE = 1024 # Number of sample in each batch
    torch_dataset = Data.TensorDataset(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float(), torch.from_numpy(tauScale_train).float())
    loader = Data.DataLoader(dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=True)

    print("Test Files:")
    xtest, ytest, tauScaletest, testMask = preprocess(testFiles, testfilemaskpercents, scaleStr, testfileUgs, testfileRes, testfileB0s, size, irun)
    auxDataDict.update(testMask) 

    mask =  np.random.rand(xtest.shape[0]) > valMaskPercent
    auxDataDict["valMask_"+str(irun)]=mask
    x_test=xtest[mask]
    x_val=xtest[~mask]
    del xtest
    y_test=ytest[mask]
    y_val=ytest[~mask]
    del ytest
    tauScale_test=tauScaletest[mask]
    tauScale_val=tauScaletest[~mask]
    del tauScaletest
    gc.collect()

    torch_dataset_test = Data.TensorDataset(torch.from_numpy(x_test).float(),torch.from_numpy(y_test).float(), torch.from_numpy(tauScale_test).float())
    loader_test = Data.DataLoader(dataset=torch_dataset_test,batch_size=BATCH_SIZE,shuffle=True)

    torch_dataset_val = Data.TensorDataset(torch.from_numpy(x_val).float(),torch.from_numpy(y_val).float(), torch.from_numpy(tauScale_val).float())
    loader_val = Data.DataLoader(dataset=torch_dataset_val,batch_size=BATCH_SIZE,shuffle=True)

    numerator=1
    LossWeights = np.array([numerator/np.std(y_train[:,i]) for i in range(y_train.shape[1])])
    #LossWeights[0] = 0.25*LossWeights[0]
    print('Lossweights:')
    print(LossWeights)
    weights=torch.from_numpy(LossWeights).to(device)


    model=CNDNN(Nhid,N=4,size=size,device=device).float().to(device)
    n_epochs = 500 # Max number of epochs
    patience = 20
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,range(1,3), gamma=0.10)
    early_stopper = EarlyStopper(patience=patience, min_delta=0)
    criterion = torch.nn.L1Loss() # MSE loss function

    validation_loss = list()
    train_loss = list()
    test_loss = list()
    min_loss = np.inf #Number of epochs

    for epoch in range(n_epochs):
        print(epoch)
        print('LR: ',print(scheduler.get_last_lr()))
        train_model(model,criterion,loader,optimizer, scheduler, weights,device)
        train_loss.append(test_model(model,criterion,loader,weights,device, 'train'))
        validation_loss.append(test_model(model,criterion,loader_val,weights,device))
        test_loss.append(test_model(model,criterion,loader_test,weights,device,'test'))
        if validation_loss[-1] < min_loss:
            torch.save(model.state_dict(),savePath+saveFile+str(irun)+'.pt',pickle_protocol=-1)
            min_loss = validation_loss[-1]
        if early_stopper.early_stop(validation_loss[-1]):
            print('ES epoch: '+str(epoch-patience))
            break
    model.load_state_dict(torch.load(savePath+saveFile+str(irun)+'.pt',map_location=device),strict=False)

    if save==1:
        torch.save(model.state_dict(),savePath+saveFile+str(irun)+'.pt',pickle_protocol=-1)

    auxDataDict["train_loss_"+str(irun)]=train_loss
    auxDataDict["validation_loss_"+str(irun)]=validation_loss
    auxDataDict["test_loss_"+str(irun)]=test_loss
    

    fig1 = plt.figure(figsize = (20, 6))
    epochs=range(len(validation_loss))
    plt.plot(epochs,train_loss, label = 'Train loss')
    plt.plot(epochs,validation_loss, label = 'Val. loss')
    plt.plot(epochs,test_loss, label = 'Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.savefig(plotPath+saveFile+str(irun)+'_lossCurve.png')

    fig1 = plt.figure(figsize = (20, 6))
    epochs=range(len(test_loss))
    plotEpochs=slice(len(test_loss)-patience-1,len(test_loss))
    plt.plot(epochs[plotEpochs],train_loss[plotEpochs], label = 'Train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.savefig(plotPath+saveFile+str(irun)+'_TrainLossCurve.png')

    fig1 = plt.figure(figsize = (20, 6))
    epochs=range(len(test_loss))
    plotEpochs=slice(len(test_loss)-2*patience-1,len(test_loss))
    plt.plot(epochs[plotEpochs],validation_loss[plotEpochs], label = 'Val. Loss')
    plt.plot(epochs[plotEpochs],test_loss[plotEpochs], label = 'Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.savefig(plotPath+saveFile+str(irun)+'_TestLossCurve.png')

    model.eval()
    
    print("Test data")
    ypred = model(torch.from_numpy(x_test).float().to(device)).cpu().detach().numpy().squeeze()
    y_pred=ypred*tauScale_test
    fig2,ax2 = plt.subplots(1,y_pred.shape[1],figsize = (20, 6))
    for i in range(y_pred.shape[1]):
        r2[irun,i]=r2_score(y_test[:,i], y_pred[:,i])
        r[irun,i]=np.corrcoef(y_test[:,i], y_pred[:,i])[0, 1]
        print("Skills for "+y_text[i])
        print("R^2: %.4f" % r2[irun,i] )
        print("Correlation: %.4f" % +r[irun,i]+"\n")

        ax2[i].scatter(y_test[:,i]*LossWeights[i], y_pred[:,i]*LossWeights[i])
        xmin,xmax=ax2[i].get_xlim()
        ymin,ymax=ax2[i].get_ylim()
        ax2[i].plot([xmin,xmax],[xmin,xmax])
        ax2[i].set_xlim([xmin,xmax])
        ax2[i].set_xlabel('True')
        ax2[i].set_ylabel('Predicted')
        ax2[i].set_title(y_text[i])
    plt.savefig(plotPath+saveFile+str(irun)+'_TestStats.png') 

    print("Validation data")
    ypred = model(torch.from_numpy(x_val).float().to(device)).cpu().detach().numpy().squeeze()
    y_pred=ypred*tauScale_val
    fig2,ax2 = plt.subplots(1,y_pred.shape[1],figsize = (20, 6))
    for i in range(y_pred.shape[1]):
        print("Skills for "+y_text[i])
        print("R^2: %.4f" % r2_score(y_val[:,i], y_pred[:,i]))
        print("Correlation: %.4f" % np.corrcoef(y_val[:,i], y_pred[:,i])[0, 1]+"\n")

        ax2[i].scatter(y_val[:,i]*LossWeights[i], y_pred[:,i]*LossWeights[i])
        xmin,xmax=ax2[i].get_xlim()
        ymin,ymax=ax2[i].get_ylim()
        ax2[i].plot([xmin,xmax],[xmin,xmax])
        ax2[i].set_xlim([xmin,xmax])
        ax2[i].set_xlabel('True')
        ax2[i].set_ylabel('Predicted')
        ax2[i].set_title(y_text[i])
    plt.savefig(plotPath+saveFile+str(irun)+'_ValDataStats.png')

    print("Train data")
    mask =  np.random.rand(x_train.shape[0]) < 0.05
    ypred = model(torch.from_numpy(x_train[mask]).float().to(device)).cpu().detach().numpy().squeeze()
    y_pred=ypred*tauScale_train[mask]
    ytrain=y_train[mask]
    fig2,ax2 = plt.subplots(1,y_pred.shape[1],figsize = (20, 6))
    for i in range(y_pred.shape[1]):
        print("Skills for "+y_text[i])
        print("R^2: %.4f" % r2_score(ytrain[:,i], y_pred[:,i]))
        print("Correlation: %.4f" % np.corrcoef(ytrain[:,i], y_pred[:,i])[0, 1]+"\n")

        ax2[i].scatter(ytrain[:,i]*LossWeights[i], y_pred[:,i]*LossWeights[i])
        xmin,xmax=ax2[i].get_xlim()
        ymin,ymax=ax2[i].get_ylim()
        ax2[i].plot([xmin,xmax],[xmin,xmax])
        ax2[i].set_xlim([xmin,xmax])
        ax2[i].set_xlabel('True')
        ax2[i].set_ylabel('Predicted')
        ax2[i].set_title(y_text[i])
    plt.savefig(plotPath+saveFile+str(irun)+'_TrainDataStats.png')

    torch.cuda.empty_cache()

print(r)
print(r2)
   
for v in range(r2.shape[1]):
        print(y_text[v]+' avg. R^2 is '+str(np.mean(r2[:,v]))+' +/- '+str(np.std(r2[:,v])))
print('Overall avg. R^2 is '+str(np.mean(r2))+' +/- '+str(np.std(np.mean(r2,axis=1))))
pickle.dump(auxDataDict,open(savePath+saveFile[:-1]+'.pkl', "wb" ))
