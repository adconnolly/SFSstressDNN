#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
#import netCDF4 as nc
import xarray as xr
from coarsen import coarsen


path='/glade/u/home/adac/work/DNSdata/DNS_SBL_L320/'
Repath='Re2700'
path=path+Repath+'/'

rcoarse=60#40 # 40 DNS grid avg window 
rzcoarse=156
rgrid=15
rzgrid=39


#timesteps=[590200,590400]
#print(timesteps)

if Repath=='Re900':
  timesteps1=np.arange(590200,596000,200)
  timesteps2=np.arange(596000,610000+1,1000)
  timesteps=np.concatenate((timesteps1,timesteps2))
elif Repath=='Re1800':
  timesteps=np.arange(616000,630000+1,1000)
elif Repath=='Re2700':
  timesteps=np.arange(1368000,1386000+1,9000)

nt=len(timesteps)

for i in range(nt):
  if i==0:
    x, y, z, ubar, vbar, wbar, t11, t22, t33, t12, t13, t23, bbar, ubbar, vbbar, wbbar, pbar =coarsen(path,timesteps[i],rcoarse,rzcoarse,rgrid,rzgrid)
    #x, y, z, ubar, vbar, wbar, t11, t22, t33, t12, t13, t23, bbar, ubbar, vbbar, wbbar, dudxbar, dudybar, dudzbar, dvdxbar, dvdybar, dvdzbar, dwdxbar, dwdybar, dwdzbar, pbar =coarsen(path,timesteps[i],rcoarse,rzcoarse,rgrid,rzgrid)

    u=np.zeros(np.append(ubar.shape,nt))
    u[:,:,:,i]=ubar
    
    v=np.zeros(np.append(ubar.shape,nt))
    v[:,:,:,i]=vbar

    w=np.zeros(np.append(ubar.shape,nt))
    w[:,:,:,i]=wbar

    tau11=np.zeros(np.append(ubar.shape,nt))
    tau11[:,:,:,i]=t11

    tau22=np.zeros(np.append(ubar.shape,nt))
    tau22[:,:,:,i]=t22

    tau33=np.zeros(np.append(ubar.shape,nt))
    tau33[:,:,:,i]=t33

    tau12=np.zeros(np.append(ubar.shape,nt))
    tau12[:,:,:,i]=t12

    tau13=np.zeros(np.append(ubar.shape,nt))
    tau13[:,:,:,i]=t13

    tau23=np.zeros(np.append(ubar.shape,nt))
    tau23[:,:,:,i]=t23

    b=np.zeros(np.append(ubar.shape,nt))
    b[:,:,:,i]=bbar

    ub=np.zeros(np.append(ubar.shape,nt))
    ub[:,:,:,i]=ubbar

    vb=np.zeros(np.append(ubar.shape,nt))
    vb[:,:,:,i]=vbbar
    
    wb=np.zeros(np.append(ubar.shape,nt))
    wb[:,:,:,i]=wbbar

    #epsilon=np.zeros(np.append(ubar.shape,nt))
    #epsilon[:,:,:,i]=eps
    
    #dudx=np.zeros(np.append(ubar.shape,nt))
    #dudx[:,:,:,i]=dudxbar

    #dudy=np.zeros(np.append(ubar.shape,nt))
    #dudy[:,:,:,i]=dudybar
 
    #dudz=np.zeros(np.append(ubar.shape,nt))
    #dudz[:,:,:,i]=dudzbar

    #dvdx=np.zeros(np.append(ubar.shape,nt))
    #dvdx[:,:,:,i]=dvdxbar

    #dvdy=np.zeros(np.append(ubar.shape,nt))
    #dvdy[:,:,:,i]=dvdybar
    
    #dvdz=np.zeros(np.append(ubar.shape,nt))
    #dvdz[:,:,:,i]=dvdzbar
    
    #dwdx=np.zeros(np.append(ubar.shape,nt))
    #dwdx[:,:,:,i]=dwdxbar

    #dwdy=np.zeros(np.append(ubar.shape,nt))
    #dwdy[:,:,:,i]=dwdybar
    
    #dwdz=np.zeros(np.append(ubar.shape,nt))
    #dwdz[:,:,:,i]=dwdzbar
    
    p=np.zeros(np.append(ubar.shape,nt))
    p[:,:,:,i]=pbar

  else:
    _,_ ,_, ubar, vbar, wbar, t11, t22, t33, t12, t13, t23, bbar, ubbar, vbbar, wbbar, pbar =coarsen(path,timesteps[i],rcoarse,rzcoarse,rgrid,rzgrid)
    #_,_ ,_, ubar, vbar, wbar, t11, t22, t33, t12, t13, t23, bbar, ubbar, vbbar, wbbar, dudxbar, dudybar, dudzbar, dvdxbar, dvdybar, dvdzbar, dwdxbar, dwdybar, dwdzbar, pbar =coarsen(path,timesteps[i],rcoarse,rzcoarse,rgrid,rzgrid)
 

    u[:,:,:,i]=ubar
    v[:,:,:,i]=vbar
    w[:,:,:,i]=wbar
    tau11[:,:,:,i]=t11
    tau22[:,:,:,i]=t22
    tau33[:,:,:,i]=t33
    tau12[:,:,:,i]=t12
    tau13[:,:,:,i]=t13
    tau23[:,:,:,i]=t23
    b[:,:,:,i]=bbar
    ub[:,:,:,i]=ubbar
    vb[:,:,:,i]=vbbar
    wb[:,:,:,i]=wbbar
    #epsilon[:,:,:,i]=eps
    #dudx[:,:,:,i]=dudxbar
    #dudy[:,:,:,i]=dudybar
    #dudz[:,:,:,i]=dudzbar
    #dvdx[:,:,:,i]=dvdxbar
    #dvdy[:,:,:,i]=dvdybar
    #dvdz[:,:,:,i]=dvdzbar
    #dwdx[:,:,:,i]=dwdxbar
    #dwdy[:,:,:,i]=dwdybar
    #dwdz[:,:,:,i]=dwdzbar
    p[:,:,:,i]=pbar


ds = xr.Dataset(
  data_vars=dict(
    u=(["z","y", "x","time"], u),
    v=(["z","y", "x","time"], v),
    w=(["z","y", "x","time"], w),
    tau11=(["z","y", "x","time"], tau11),
    tau22=(["z","y", "x","time"], tau22),
    tau33=(["z","y", "x","time"], tau33),
    tau12=(["z","y", "x","time"], tau12),
    tau13=(["z","y", "x","time"], tau13),
    tau23=(["z","y", "x","time"], tau23),
    b=(["z","y", "x","time"], b),
    ub=(["z","y", "x","time"], ub),
    vb=(["z","y", "x","time"], vb),
    wb=(["z","y", "x","time"], wb),
    #epsilon=(["z","y", "x","time"], epsilon),
    #dudx=(["z","y", "x","time"], dudx),
    #dudy=(["z","y", "x","time"], dudy),
    #dudz=(["z","y", "x","time"], dudz),
    #dvdx=(["z","y", "x","time"], dvdx),
    #dvdy=(["z","y", "x","time"], dvdy),
    #dvdz=(["z","y", "x","time"], dvdz),
    #dwdx=(["z","y", "x","time"], dwdx),
    #dwdy=(["z","y", "x","time"], dwdy),
    #dwdz=(["z","y", "x","time"], dwdz),
    p=(["z","y", "x","time"], p),
    ),
  coords=dict(
    z=(["z"],z),
    y=(["y"],y),
    x=(["x"],x),
    time=(["time"],timesteps),
    ),
)

ds.to_netcdf("coarse"+str(int(rcoarse/rgrid))+"x"+str(rgrid)+str(rzgrid)+"_"+Repath+".nc")
print(ds)
