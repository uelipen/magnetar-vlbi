import astropy.io.fits as fits
import numpy as np
from pylab import *
import matplotlib.pylab as plt
from scipy import special

antenna='pt'
session='B'
#session=''
source='J1745-2900'

if (session == 'C' ):
   pulses1=np.load('NP/scan29.npy')
   pulses2=np.load('NP/scan30.npy')
   pulses3=np.load('NP/scan31.npy')
   pulses3=np.roll(pulses3,1,axis=0)
   pulses4=np.load('NP/scan32.npy')
   #pulses4=np.roll(pulses4,1,axis=0)
   pulses5=np.load('NP/scan33.npy')
   pulses5=np.roll(pulses5,1,axis=0)
   pulses6=np.load('NP/scan34.npy')
   pulses6=np.roll(pulses6,1,axis=0)
   pulses7=np.load('NP/scan35.npy')
   pulses7=np.roll(pulses7,1,axis=0)
   pulses8=np.load('NP/scan36.npy')
   pulses8=np.roll(pulses8,1,axis=0)
   pulses9=np.load('NP/scan37.npy')
   pulses9=np.roll(pulses9,1,axis=0)
   pulses10=np.load('NP/scan38.npy')
   #pulses10=np.roll(pulses10,1,axis=0)
   pulses11=np.load('NP/scan39.npy')
   #pulses11=np.roll(pulses11,1,axis=0)
   pulses12=np.load('NP/scan40.npy')
   pulses12=np.roll(pulses12,1,axis=0)
   pulses13=np.load('NP/scan41.npy')
   pulses14=np.load('NP/scan42.npy')
   pulses=np.vstack([pulses1,pulses2[:,:],pulses3,pulses4,pulses4[-3:-1,:],pulses5,pulses6,pulses6[-2:-1,:],pulses7,pulses8,pulses9,pulses9[-3:-1,:],pulses10,pulses10[-2:-1,:],pulses11,pulses12,pulses12[-2:-1,:],pulses13,pulses13[-2:-1,:],pulses14])
if (session == 'B' ):
   pulses1=np.load('NP/scan15.npy')
   pulses2=np.load('NP/scan16.npy')
   pulses3=np.load('NP/scan17.npy')
   pulses4=np.load('NP/scan18.npy')
   pulses4=np.roll(pulses4,1,axis=0)
   pulses5=np.load('NP/scan19.npy')
   pulses6=np.load('NP/scan20.npy')
   pulses7=np.load('NP/scan21.npy')
   pulses8=np.load('NP/scan22.npy')
   pulses8=np.roll(pulses8,1,axis=0)
   pulses9=np.load('NP/scan23.npy')
   pulses9=np.roll(pulses9,2,axis=0)
   pulses10=np.load('NP/scan24.npy')
   pulses10=np.roll(pulses10,1,axis=0)
   pulses11=np.load('NP/scan25.npy')
   pulses11=np.roll(pulses11,1,axis=0)
   pulses12=np.load('NP/scan26.npy')
   pulses12=np.roll(pulses12,1,axis=0)
   pulses13=np.load('NP/scan27.npy')
   pulses14=np.load('NP/scan28.npy')
   pulses=np.vstack([pulses1,pulses2[:,:],pulses2[-2:-1,:],pulses3,pulses4,pulses4[-3:-1,:],pulses5,pulses6,pulses6[-2:-1,:],pulses7,pulses8,pulses9,pulses9[-3:-1,:],pulses10,pulses10[-2:-1,:],pulses11,pulses12,pulses12[-2:-1,:],pulses13,pulses13[-2:-1,:],pulses14])
if (session == '' ) :
   pulses1=np.load('NP/scan1.npy')
   pulses2=np.load('NP/scan2.npy')
   pulses3=np.load('NP/scan3.npy')
   pulses4=np.load('NP/scan4.npy')
   pulses4=np.roll(pulses4,1,axis=0)
   pulses5=np.load('NP/scan5.npy')
   pulses6=np.load('NP/scan6.npy')
   pulses7=np.load('NP/scan7.npy')
   pulses8=np.load('NP/scan8.npy')
   pulses9=np.load('NP/scan9.npy')
   pulses9=np.roll(pulses9,1,axis=0)
   pulses10=np.load('NP/scan10.npy')
   pulses11=np.load('NP/scan11.npy')
   pulses12=np.load('NP/scan12.npy')
   pulses13=np.load('NP/scan13.npy')
   pulses14=np.load('NP/scan14.npy')
   pulses=np.vstack([pulses1,pulses2[1:,:],pulses2[-2:-1,:],pulses3,pulses4,pulses4[-3:-1,:],pulses5,pulses6,pulses6[-2:-1,:],pulses7,pulses8,pulses9,pulses9[-3:-1,:],pulses10,pulses10[-2:-1,:],pulses11,pulses12,pulses12[-2:-1,:],pulses13,pulses13[-2:-1,:],pulses14])

nt=pulses.shape[0]
print nt
times=np.load('time'+antenna+'y'+source+session+'.npy')
dt=times-times[0]
dt1=dt/dt[1]
tn=times[:nt]
for ipol in range(2) :
   visc0=np.load('pty'+antenna+'y'+source+session+str(ipol)+'.npy')
   visc=np.zeros([42,nt,1024]).astype('c8')
   for i in range(nt) :
       j=i
       if(session=='' and (i==593 or i==1067 or i==1541)) :
           j=j+1
       if(session=='B' and np.any(i==np.array([277,435,593,1225,1699,1857]))) :
           j=j+1
       if(session=='C' and np.any(i==np.array([435,751,2015,1699]))) :
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
   
   visc[:,:,640:768]=0 # bad IF
   if (session == '') :
   	visc[:,435,:]=0
        visc[:,1383,:]=0
        visc[:,1225,:]=0
   if (session == 'B') :
   	visc[:,1541,:]=0
   if (session == 'C') :
   	visc[:,1541,:]=0
   	visc[:,1782,:]=0
   
   #visc[:,:,:512]=0
   
   tmp=np.load('J1751model'+antenna+'ypol'+str(ipol)+session+'.npy')
   visc=visc*tmp[np.newaxis,:nt,:]
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
   shift=(np.round((pulses.argmax(axis=1)-1024)/0.0025/2048))
   shift0=pulses.argmax(axis=1)
   shift0[shift0==0]=pulses.shape[1]/2
   weights=pulses.max(axis=1)
   for i in range(pulses.shape[0]) :
	weights[i]=pulses[i,shift0[i]-5:shift0[i]+5].mean()

   weights=weights+1.e-10
   
   visc=visc*weights[np.newaxis,:,np.newaxis]
   
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
   cal=abs(lagmean[gatecal,nlagcal])/(lagmean[gatecal,nlagcal]+1.e-30)
   lagcal=lagmean*cal
   imshow(np.abs(lagmean[2:41,2000:2100]),interpolation='nearest')
   
   imshow(np.real(lagcal[2:41,2000:2100]),interpolation='nearest')
   
   plt.figure()
   plt.plot(np.real(lagcal[:,nlagcal]))
   plt.plot(np.imag(lagcal[:,nlagcal]))
   
   lag1=lag/sqrt(np.abs(lag)+1.e-30)
   
   vis2=np.reshape(visc.mean(axis=2),(42,41,53)).mean(axis=2)
   if ( session == '' ) :
   	vis2[:,38]=0
   if ( session == 'C' ) :
   	vis2[:,11]=0
   	vis2[:,23]=0
   dt=arange(vis2.shape[0])-13.5
   dts=np.sign(dt)*np.sqrt(np.abs(dt))
   ampl=0.00001
   sigma2=3**2
   calweight=exp(-abs(dts)**2/sigma2)*ampl
   if (antenna=='la') :
       calweight=exp(-abs(dts)**2/sigma2)*ampl*special.jn(0,abs(dts/0.9))
	
   calweight[:13]=0
   calweight=calweight/sum(calweight)
   cal=vis2[13:15,:].mean(axis=0)
   cal=(vis2*calweight[:,np.newaxis]).mean(axis=0)
   #cal=vis2[17:20,:].mean(axis=0)
   #cal=vis2[9,:].mean(axis=0).mean(axis=0)
   cal=abs(cal)/(cal+1.e-20)
   #cal[:26]=1
   #cal=1
   viscal=vis2*cal
   viscm=visc.mean(axis=2)/(weights+1.e-30)

   if (ipol == 0) : 
	visall=vis2
	viscalall=viscal
	viscmall=viscm
	

viscalall=viscalall+viscal
visall=visall+vis2

calweight=exp(-abs(dts)**2/sigma2)*ampl
if (antenna=='la') :
    calweight=exp(-abs(dts)**2/sigma2)*ampl*special.jn(0,abs(dts*1.3/0.9))

calweight[:13]=0
calweight=calweight/sum(calweight)
cal=(visall*calweight[:,np.newaxis]).mean(axis=0)
cal=abs(cal)/(cal+1.e-20)
viscalall=visall*cal
weights1=np.reshape(weights,(41,53)).sum(axis=1)
viscmall=viscmall+viscm
np.save('gatedvis'+antenna+session+'.npy',(visall,viscalall,weights1))
if (antenna=='pt'):
  np.save('pulses'+antenna+session+'.npy',(viscmall,weights,cal))



vism=(vis2*cal)[:,26:].mean(axis=1)
vism=(visall)[:,25:].mean(axis=1)
vism=(viscalall[:,12:24]).mean(axis=1)
vism=(viscalall[:,26:]).mean(axis=1)
vism=(viscalall[:,5:20]).mean(axis=1)
ampl=0.00001
plt.figure()
plt.plot(dts,real(vism))
plt.plot(dts,imag(vism))
plt.plot(dts,exp(-abs(dts)**2/sigma2)*ampl)
plt.plot(dts,exp(-abs(dts)**2/sigma2)*ampl*special.jn(0,abs(dts/0.9)))

plt.figure()
imshow(np.real(viscalall),interpolation='nearest')
plt.colorbar()
plt.figure()
imshow(np.imag(viscalall),interpolation='nearest')
plt.colorbar()
plt.figure()

plt.figure()
imshow(np.real(visall),interpolation='nearest')
plt.colorbar()
plt.figure()
imshow(np.imag(visall),interpolation='nearest')
plt.colorbar()

plt.figure()
plt.plot(np.angle(cal),'x')

