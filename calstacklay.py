import astropy.io.fits as fits
import numpy as np
from pylab import *
import matplotlib.pylab as plt


pulses1=np.load('NP/scan1.npy')
pulses2=np.load('NP/scan2.npy')
pulses3=np.load('NP/scan3.npy')
pulses4=np.load('NP/scan4.npy')/1000
pulses5=np.load('NP/scan5.npy')
pulses6=np.load('NP/scan6.npy')
pulses7=np.load('NP/scan7.npy')
pulses8=np.load('NP/scan8.npy')
pulses9=np.load('NP/scan9.npy')/1000
pulses10=np.load('NP/scan10.npy')
pulses11=np.load('NP/scan11.npy')
pulses12=np.load('NP/scan12.npy')
pulses13=np.load('NP/scan13.npy')
pulses14=np.load('NP/scan14.npy')
pulses=np.vstack([pulses1,pulses2[1:,:],pulses2[-2:-1,:],pulses3,pulses4,pulses4[-3:-1,:],pulses5,pulses6,pulses6[-2:-1,:],pulses7,pulses8,pulses9,pulses9[-3:-1,:],pulses10,pulses10[-2:-1,:],pulses11,pulses12,pulses12[-2:-1,:],pulses13,pulses13[-2:-1,:],pulses14])

nt=pulses.shape[0]
print nt
visc0=np.load('ptyla0.npy')
times=np.load('times.npy')
dt=times-times[0]
dt1=dt/dt[1]
tn=times[:nt]
visc=np.zeros([42,nt,1024]).astype('c8')
for i in range(nt) :
    j=i
    if(i==593 or i==1067 or i==1541) :
        j=j+1
    visc[:,i,:]=visc0[np.where(times==tn[j]),:]

visc[:,:,:10]=0
visc[:,:,108]=0
visc[:,:,129:131]=0
visc[:,:,346]=0
visc[:,:,161:170]=0
visc[:,:,148:153]=0
visc[:,1680:2020,689:706]=0
visc[:,1680:2020,714:721]=0
visc[:,1680:2020,741:749]=0
visc[:,:250,657:674]=0
visc[:,:250,707:714]=0
visc[:,:250,749:756]=0
visc[:,:600,126:132]=0
visc[:,508:515,301:303]=0
visc[:,508:515,301:303]=0
visc[:,245:460:,300:315]=0
visc=np.reshape(visc,(42,nt,8,128))
visc=np.reshape(visc[:,:,:,-1::-1],(42,nt,8*128))

tmp=np.load('sgramodellaypol0.npy')
visc=visc*tmp[np.newaxis,:,:]
imshow(np.abs(visc[20,:,:].T),interpolation='nearest')
vis1=visc/sqrt(np.abs(visc)+1.e-20)
vismean=(visc[1,:,:]/0.05+visc[41,:,:]/0.0625)/2
visc=visc-vismean[np.newaxis,:,:]*0.0025

lag=np.fft.fftshift(np.fft.fft(visc,axis=2),axes=(2,))
imshow(np.abs(lag[20,:,:].T),interpolation='nearest')
lag2=lag[2:41,:,490:530]
lag2=lag2.transpose((1,2,0))
lag2=np.reshape(lag2,(nt,39*40))
imshow(np.abs(lag2.T),interpolation='nearest')

pulses[:,:800]=0
pulses[:,1250:]=0
weights=pulses.max(axis=1)
shift=(np.round((pulses.argmax(axis=1)-1024)/0.0025/2048))
#visc=visc[:,:nt-1,:]*weights[np.newaxis,1:,np.newaxis]
#visc=visc[:,1:,:]*weights[np.newaxis,:nt-1,np.newaxis]

visc=visc*weights[np.newaxis,:,np.newaxis]

#visc=visc[:,3:,:]*weights[np.newaxis,:nt-3,np.newaxis]
#visc=visc[:,:nt-2,:]*weights[np.newaxis,2:,np.newaxis]

visc[:2,:,:]=0
visc[41:,:,:]=0
for i in range(visc.shape[1]) :
     visc[:,i,:]=np.roll(visc[:,i,:],int(-shift[i]),axis=0)

vistime=visc[:,2100:,:].mean(axis=1)
vs=vistime.shape
vispad=np.zeros((vs[0],vs[1]*4),dtype=complex)
vispad[:,:vs[1]]=vistime
lagmean=np.fft.fftshift(np.fft.fft(vispad,axis=1),axes=(1,))
gatecal=13
nlagcal=2049 # 2050
cal=abs(lagmean[gatecal,nlagcal])/lagmean[gatecal,nlagcal]
lagcal=lagmean*cal
imshow(np.abs(lagmean[2:41,2000:2100]),interpolation='nearest')

imshow(np.real(lagcal[2:41,2000:2100]),interpolation='nearest')

plt.figure()
plt.plot(np.real(lagcal[:,nlagcal]))
plt.plot(np.imag(lagcal[:,nlagcal]))

lag1=lag/sqrt(np.abs(lag))


vis2=np.reshape(visc,(42,41,53,8,128)).mean(axis=4).mean(axis=2)
vis2=np.reshape(visc.mean(axis=2),(42,41,53)).mean(axis=2)
vis2[:,38]=0
run svd.py
vis2c=svd_model(vis2)
vism=vis2c[:,25:].mean(axis=1)
plt.figure()
plt.plot(real(vism))
plt.plot(imag(vism))


