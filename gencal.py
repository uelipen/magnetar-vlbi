import astropy.io.fits as fits
import numpy as np
from pylab import *
import matplotlib.pylab as plt

nt=2173
session='B'
#session=''
source='J1745-2900'+session
for antenna in ['ov'] :
  (w,omega0,ft)=np.load('J1751modelsvd'+antenna+'ypol'+str(0)+session+'.npy')
  (w,omega1,ft)=np.load('J1751modelsvd'+antenna+'ypol'+str(1)+session+'.npy')
  times=np.load('time'+antenna+'y'+source+'.npy')
  omegam=(omega0+omega1)/2
  timeref=times.min()*0.125+times.max()*0.875
  print omegam,timeref,times.min(),times.max(), times.max()-times.min()
  for ipol in range(2) : 
    (w,omega,ft)=np.load('J1751modelsvd'+antenna+'ypol'+str(ipol)+session+'.npy')
    if (session == 'C1' and antenna == 'pt') :
	omegam=omega
    pref=exp((0.+1.j)*timeref*(omega-omegam))
    tmodel=ft*exp((0.+1.j)*times[:nt,np.newaxis]*omegam)*pref*w
    np.save('J1751model'+antenna+'ypol'+str(ipol)+session+'.npy',tmodel)
    print omega
