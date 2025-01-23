import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

var="u"

Re_all=['Re900/','Re1800/','Re2700/']
timesteps_all=[[600000],[623000],[1377000]]
outdir=""


for iRe in range(len(Re_all)):
  Re=Re_all[iRe]
  path='/burg/glab/projects/DNS_SBL_L320/'+Re
  timesteps=timesteps_all[iRe]
  #timesteps1=np.arange(590200,596000,200)
  #timesteps2=np.arange(596000,610000+1,1000)
  #timesteps=np.concatenate((timesteps1,timesteps2))
  nt=len(timesteps)

  for i in range(nt):

    ds = xr.open_dataset(path+var+str(timesteps[i])+'.nc',decode_times=0)
    print(ds)
    
    w=ds[var][0].values

    try:
      x=ds['xh'].values
    except:
      x=ds['x'].values
    nx=len(x)
    dx=np.mean(np.diff(x))

    try:
      y=ds['yh'].values
    except:
      y=ds['y'].values
    ny=len(y)
  
    try:
      z=ds['zh'].values
    except:
      z=ds['z'].values

    hgt=10.
    k=[(np.abs(z - hgt)).argmin()]

    kxy = 2*np.pi*np.fft.rfftfreq(nx,d=dx)
    dk=kxy[1] # should be smallest wavenumber 2pi/(length_fline*dx)
    lenf=len(kxy)# length_fline/2+1
    sp = np.empty([ny,lenf])
    #sp = np.empty([nF,avgt,length_yline,lenf])
    #sP = np.empty([nF,lenf])
    print('Plot spectra plot for time '+str(timesteps[i])+' at height '+str(z[k]))

    
    #Different detrending options
    #N = np.linspace(0,Wall[F,i,0,iy]-Wall[F,i,-1,iy],Wall[F,i,:,iy].shape[0])
    #N = -np.mean(Wall[F,i,:,iy])
    N=0 # detrending not necessary b/c periodic BCs
    for j in range(ny):
      sp_tmp = np.fft.rfft(w[k,j,:]+N)
      sp[j,:] = (np.power(sp_tmp.real,2) + np.power(sp_tmp.imag,2))*dx/2.0/np.pi/nx #missing factor of two from Durran paper neccesary for the parseval's thereom to work
      #sp[j,-1] = sp[j,-1]/2.0 #Nyquist frequency (last entry by order of reqdiag) is unique for even pts FFT, so has twice the power
      sp[j,0] = sp[j,0]/2.0 # Calculating the 0 wavenumber, won't be plotted on a log log but just to check the assumption of zero power at zero wavenumber for different detrending methods
      #wsqr=np.sum(np.power(w[k,j],2))/nx/2.0
      #wsqr_fft=np.sum(sp[j])*dk
      #print(wsqr,wsqr_fft)
      
    sP = np.mean(sp,axis=0)
    power = abs(sP)
  
    ds_current = xr.Dataset(
    data_vars=dict(
      power=(["k"+Re[:-1]], power) ),
    coords=dict(
      k=(["k"+Re[:-1]],kxy)  ))

    ds_current.to_netcdf(outdir+Re[:-1]+str(timesteps[i])+".nc")
    del ds_current
