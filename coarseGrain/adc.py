#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
#import netCDF4 as nc
#import xarray as xr


def topHatFilter(u_nopad,zwindow,ywindow,xwindow,nz,ny,nx,dz_nopad,dy_nopad,dx_nopad):

  nxLES=int(len(dx_nopad)/nx)
  nyLES=int(len(dy_nopad)/ny)
  nzLES=int(len(dz_nopad)/nz)
  u=np.pad(u_nopad,((0,zwindow-nz),(ywindow//2-ny//2,ywindow//2-ny//2),(xwindow//2-nx//2,xwindow//2-nx//2)),mode='wrap')
  u[u.shape[0]-(zwindow-nz):,:,:]=0

  ubar=np.zeros([nzLES,nyLES,nxLES])
  dy=np.pad(dy_nopad,(ywindow//2-ny//2,ywindow//2-ny//2),mode='wrap')
  dx=np.pad(dx_nopad,(xwindow//2-nx//2,xwindow//2-nx//2),mode='wrap')
  dz=np.pad(dz_nopad,(0,zwindow-nz),mode='wrap')
  for k in range(nzLES):
    uavgz=np.average(u[k*nz:k*nz+zwindow],axis=0,weights=dz[k*nz:k*nz+zwindow])
    for j in range(nyLES):
      uavgzy=np.average(uavgz[j*ny:j*ny+ywindow],axis=0,weights=dy[j*ny:j*ny+ywindow])
      for i in range(nxLES):
        ubar[k,j,i]=np.average(uavgzy[i*nx:i*nx+xwindow],weights=dx[i*nx:i*nx+xwindow])

  return ubar

def topHatFilterGrid(x_nopad,xwindow,nx,dx_nopad):

  nxLES=int(len(dx_nopad)/nx)
  x=np.pad(x_nopad,(xwindow//2-nx//2,xwindow//2-nx//2),mode='wrap')
  x[:xwindow//2-nx//2]=x[:xwindow//2-nx//2]-(x_nopad[-1]+dx_nopad[-1]/2.0)
  x[len(x)-(xwindow//2-nx//2):]=x[len(x)-(xwindow//2-nx//2):]+(x_nopad[-1]+dx_nopad[-1]/2.0)
  dx=np.pad(dx_nopad,(xwindow//2-nx//2,xwindow//2-nx//2),mode='wrap')
  xcoarse=[np.average(x[i*nx:i*nx+xwindow],weights=dx[i*nx:i*nx+xwindow]) for i in range(nxLES)]
  
  return xcoarse

def topHatFilterZGrid(z_nopad,zwindow,nz,dz_nopad):

  nzLES=int(len(dz_nopad)/nz)
  z=np.pad(z_nopad,(0,zwindow-nz),mode='wrap')
  z[len(z)-(zwindow-nz):]=z[len(z)-(zwindow-nz):]+(z_nopad[-1]+dz_nopad[-1]/2.0)
  dz=np.pad(dz_nopad,(0,zwindow-nz),mode='wrap')
  zcoarse=[np.average(z[i*nz:i*nz+zwindow],weights=dz[i*nz:i*nz+zwindow]) for i in range(nzLES)]

  return zcoarse

def topHatFilter11(u,zwindow,ywindow,xwindow,dz,dy,dx):

  nxLES=int(len(dx)/xwindow)
  nyLES=int(len(dy)/ywindow)
  nzLES=int(len(dz)/zwindow)

  ubar=np.zeros([nzLES,nyLES,nxLES])

  for k in range(nzLES):
    uavgz=np.average(u[k*zwindow:(k+1)*zwindow],axis=0,weights=dz[k*zwindow:(k+1)*zwindow])
    for j in range(nyLES):
      uavgzy=np.average(uavgz[j*ywindow:(j+1)*ywindow],axis=0,weights=dy[j*ywindow:(j+1)*ywindow])
      for i in range(nxLES):
        ubar[k,j,i]=np.average(uavgzy[i*xwindow:(i+1)*xwindow],weights=dx[i*xwindow:(i+1)*xwindow])

  return ubar

###### Strangeness of nonuniform grid with sparse unique values for dx
#dxh=np.diff(xh)
#dx=np.diff(x)
#print(np.unique(dx).shape)
#print(np.mean(dx))
#print(np.min(dx))
#print(np.max(dx))
#print(np.unique(dx).shape) # 10 unique values for xh, maybe do to ifft of pseudospectral method to physical space?
#print(dx[-30:]) # there is a pattern in the last 30 values which will allow me to predict the lastgrid spacing for periodicity 
#dx=np.append(dx,(dx[-2]+dx[0])/2) # grid is only nearly uniform, not sure if should use first values and go backward or last values and go forward to predict the periodic grid cell

### looping through and doing trapz is alternate to destaggering and straight averaging
#ncoarse=int(len(x)/rcoarse)
#uLES=np.zeros([u.shape[0],u.shape[1],ncoarse])
#for i in range(ncoarse):
#  uLES[:,:,i]=np.trapz(u[:,:,i*(rcoarse+1):(i+1)*(rcoarse+1)+1],xh[i*(rcoarse+1):(i+1)*(rcoarse+1)+1])

