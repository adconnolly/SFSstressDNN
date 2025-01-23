#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
#import netCDF4 as nc
import xarray as xr
import gc
from adc import *


def coarsen(path,timestep_numeric,rcoarse,rzcoarse,rgrid,rzgrid):

  timestep=str(timestep_numeric)

  uds = xr.open_dataset(path+'u'+timestep+'.nc',decode_times=0)
  vds = xr.open_dataset(path+'v'+timestep+'.nc',decode_times=0)
  wds = xr.open_dataset(path+'w'+timestep+'.nc',decode_times=0)
  bds = xr.open_dataset(path+'b'+timestep+'.nc',decode_times=0)
  pds = xr.open_dataset(path+'p'+timestep+'.nc',decode_times=0)
 
  #De-staggering variables
  vda=vds['v'][0][:][:][:]
  uda=uds['u'][0][:][:][:]
  wda=wds['w'][0][:][:][:]
  bda=bds['b'][0][:][:][:]
  pda=pds['p'][0][:][:][:]
  u=uda.values.astype('float32')
  uwrap=np.concatenate((u,u[:,:,0:1]),axis=2)
  u=(uwrap[:,:,:-1]+uwrap[:,:,1:])/2.0
  del uwrap
  gc.collect()

  v=vda.values.astype('float32')
  vwrap=np.concatenate((v,v[:,0:1,:]),axis=1)
  v=(vwrap[:,:-1,:]+vwrap[:,1:,:])/2.0
  del vwrap
  gc.collect()

  w=wda.values.astype('float32')
  #shape=wda.values.shape
  wpad=np.concatenate((w,np.zeros([1,w.shape[1],w.shape[2]],dtype='float32')),axis=0)
  w=(wpad[:-1,:,:]+wpad[1:,:,:])/2.0
  del wpad
  gc.collect()
  b=bda.values.astype('float32')
  p=pda.values.astype('float32')
  
  
  x=wda['x'].values #dimension of coordinate doesn't match the velocity component so that it is already unstaggered
  y=wda['y'].values
  z=uda['z'].values
  xh=uda['xh'].values #staggered values, needed to compute grid sizes
  yh=vda['yh'].values
  zh=wda['zh'].values
  xhwrap=np.append(xh,2.0*x[-1]-xh[-1]) # from xh + 2*(x-xh)=xh+2*((xh+dx/2)-xh) which assume staggered values are at grid centers, i.e. x =xh+dx/2
  yhwrap=np.append(yh,2.0*x[-1]-xh[-1])
  zhpad=np.append(zh,2.0*z[-1]-zh[-1])
  dx=np.diff(xhwrap)
  dy=np.diff(yhwrap)
  dz=np.diff(zhpad)


  #dudx=np.diff(uwrap,axis=2)/dx #uwrap is stored at edges, so diffing get gradient at centers

  #dvdx=np.append(np.diff(v,axis=2),v[:,:,0:1]-v[:,:,-1:],axis=2)/dx # because v is already stored at centers, diffing results in stored at edges, so need to append the periodic wrap around edge
  #dvdx=np.append(dvdx[:,:,-1:]+dvdx[:,:,0:1],dvdx[:,:,:-1]+dvdx[:,:,1:],axis=2)/2.0 #average gradients back to grid centers

  #dwdx=np.append(np.diff(w,axis=2),w[:,:,0:1]-w[:,:,-1:],axis=2)/dx # because v is already stored at centers, diffing results in stored at edges, so need to append the periodic wrap around edge
  #dwdx=np.append(dwdx[:,:,-1:]+dwdx[:,:,0:1],dwdx[:,:,:-1]+dwdx[:,:,1:],axis=2)/2.0#(dwdx[:,:,:-1]+dwdx[:,:,1:])/2.0 #average gradients back to grid centers
  
  #gc.collect()

  #dvdy=np.diff(vwrap,axis=1)/dy #uwrap is stored at edges, so diffing get gradient at centers

  #dudy=np.append(np.diff(u,axis=1),u[:,0:1,:]-u[:,-1:,:],axis=1)/dy # because v is already stored at centers, diffing results in stored at edges, so need to append the periodic wrap around edge
  #dudy=np.append(dudy[:,-1:,:]+dudy[:,0:1,:],dudy[:,:-1,:]+dudy[:,1:,:],axis=1)/2.0 #average gradients back to grid centers
  
  #dwdy=np.append(np.diff(w,axis=1),w[:,0:1,:]-w[:,-1:,:],axis=1)/dy # because v is already stored at centers, diffing results in stored at edges, so need to append the periodic wrap around edge
  #dwdy=np.append(dwdy[:,-1:,:]+dwdy[:,0:1,:],dwdy[:,:-1,:]+dwdy[:,1:,:],axis=1)/2.0 #average gradients back to grid centers


  #gc.collect()

  #dwdz=(np.diff(wpad,axis=0).T/dz).T #uwrap is stored at edges, so diffing get gradient at centers
  
  #dudz=np.gradient(u,z,axis=0)
  
  ##dudz=(np.append(np.diff(u,axis=0),np.zeros([1,u.shape[1],u.shape[2]]),axis=0).T/dz).T # because v is already stored at centers, diffing results in stored at edges, so need to append the periodic wrap around edge
  ##dudz=(dudz[:-1]+dudz[1:])/2.0 #average gradients back to grid centers
  
  #dvdz=np.gradient(v,z,axis=0)
  ##dvdz=(np.append(np.diff(v,axis=0),np.zeros([1,v.shape[1],v.shape[2]]),axis=0).T/dz).T # because v is already stored at centers, diffing results in stored at edges, so need to append the periodic wrap around edge
  ##dvdz=(dvdz[:-1]+dvdz[1:])/2.0 #average gradients back to grid centers

  #nx=len(x)
  #ny=len(y)
  #nz=len(z)
 
  uubar=topHatFilter(u*u,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  vvbar=topHatFilter(v*v,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  wwbar=topHatFilter(w*w,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  uvbar=topHatFilter(u*v,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  uwbar=topHatFilter(u*w,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  vwbar=topHatFilter(v*w,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)

  ubar=topHatFilter(u,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  vbar=topHatFilter(v,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  wbar=topHatFilter(w,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  bbar=topHatFilter(b,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  ubbar=topHatFilter(u*b,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  vbbar=topHatFilter(v*b,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  wbbar=topHatFilter(w*b,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)

  #epsilon=2.0*topHatFilter(dudx*dudx,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  #epsilon+=topHatFilter(dvdx*dvdx,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  #epsilon+=topHatFilter(dwdx*dwdx,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  #epsilon+=topHatFilter(dudy*dudy,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  #epsilon+=2.0*topHatFilter(dvdy*dvdy,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  #epsilon+=topHatFilter(dwdy*dwdy,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  #epsilon+=topHatFilter(dudz*dudz,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  #epsilon+=topHatFilter(dvdz*dvdz,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  #epsilon+=2.0*topHatFilter(dwdz*dwdz,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  #epsilon+=2.0*topHatFilter(dudy*dvdx,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  #epsilon+=2.0*topHatFilter(dudz*dwdx,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  #epsilon+=2.0*topHatFilter(dvdz*dwdy,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)
  pbar=topHatFilter(p,rzcoarse,rcoarse,rcoarse,rzgrid,rgrid,rgrid,dz,dy,dx)

  xcoarse=topHatFilterGrid(x,rcoarse,rgrid,dx)
  ycoarse=topHatFilterGrid(y,rcoarse,rgrid,dy)
  zcoarse=topHatFilterZGrid(z,rzcoarse,rzgrid,dz)

  tau11=uubar-ubar*ubar
  tau22=vvbar-vbar*vbar
  tau33=wwbar-wbar*wbar
  tau12=uvbar-ubar*vbar
  tau13=uwbar-ubar*wbar
  tau23=vwbar-vbar*wbar


  return xcoarse, ycoarse, zcoarse, ubar, vbar, wbar, tau11, tau22, tau33, tau12, tau13, tau23, bbar, ubbar, vbbar, wbbar, pbar

  #return xcoarse, ycoarse, zcoarse, ubar, vbar, wbar, tau11, tau22, tau33, tau12, tau13, tau23, bbar, ubbar, vbbar, wbbar, dudxbar, dudybar, dudzbar, dvdxbar, dvdybar, dvdzbar, dwdxbar, dwdybar, dwdzbar, pbar


# # compare to some straight means using the coarsen method
# uavgx=uda.coarsen(xh=rcoarse).mean()
# uavgxy=uavgx.coarsen(y=rcoarse).mean()
# uLES=uavgxy.coarsen(z=zcoarse).mean()
# print(np.mean(uLES.values-ubar))

# sanity check plots
# fig,ax=plt.subplots(1,2)
# kcoarse=5
# ax[0].contourf(x,y,u[int((kcoarse+.5)*zcoarse)][:][:],vmin=-0.0,vmax=0.03,cmap='Reds')
# ax[1].contourf(xcoarse,ycoarse,ubar[kcoarse][:][:],vmin=-0.0,vmax=0.03,cmap='Reds')
# cb=fig.colorbar(ax[1])
# cb.set_label(label='u velocity [m s$^{-1}$]')#,size=14)
# plt.savefig('ubar_eg')