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
import gc
from util import *

size=3

savePath='./'
loadPath ='../CN_paperRuns/trainedModels/'

#scaleStr='local'
#loadFile='C4_midGridReInterp_'+scaleStr+'_'
#trainFiles=["coarse4x1026_Re900.nc","coarse4x3078_Re2700.nc"]
#irun=1
#for n in range(len(trainFiles)):
#    isep = trainFiles[n].index('_')
#    loadFile=loadFile+trainFiles[n][6:isep]+trainFiles[n][isep+1:-3]+'_'
#loadFile=loadFile+str(irun)

loadFile='C4_midGridReExtrap_local_4x1026Re900_4x2052Re1800_1'
print(loadFile)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Nhid = [512,256,128,64] # Multiplicity of features in hidden layers associated with the regular representation 
model=CNDNN(Nhid,N=4,size=size,device=device).float().to(device)
model.load_state_dict(torch.load(loadPath+loadFile+'.pt',map_location=device),strict=False)

nvars=4 # u,v,w,b Even in ablation studies b/c b replaced with noise
zsize=3 # 3 vertical levels even when using 5x5 input box widths
example=torch.zeros(1,nvars*zsize,size,size).float().to(device)
traced_model=torch.jit.trace(model,example)

print(model(example))
print(traced_model(example))

example=torch.rand(2,nvars*zsize,size,size).float().to(device)
print(model(example))
print(traced_model(example))

traced_model.save(savePath+loadFile+".pt")