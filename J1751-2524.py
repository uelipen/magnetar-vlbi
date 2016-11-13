import astropy.io.fits as fits
import numpy as np
from pylab import *
import matplotlib.pylab as plt



def svd_model(arr, phase_only=True):
   """                                                                                                                          
   Take time/freq visibilities SVD, zero out all but the largest                                                                
   mode, multiply original data by complex conjugate                                                                            

   Parameters                                                                                                                   
   ----------                                                                                                                   
   arr : array_like                                                                                                             
      Time/freq visiblity matrix                                                                                                

   Returns                                                                                                                      
   -------                                                                                                                      
   Original data array multiplied by the largest SVD mode conjugate                                                             
   """

   u,s,w = np.linalg.svd(arr)
   s[4:] = 0.0
   S = np.zeros([len(u), len(w)], np.complex128)
   S[:len(s), :len(s)] = np.diag(s)


   model = np.dot(np.dot(u, S), w)

   if phase_only==True:
       return arr * np.exp(-1j * np.angle(model))
   else:
       return arr / (model)


nrfft=32
ipol=0
for antenna in ['fd','la'] :
    visc0=np.load('pty'+antenna+'yJ1751-2524.npy')
    times=np.load('time'+antenna+'yJ1751-2524.npy')
    visc=np.reshape(visc0[:,:,ipol],(visc0.shape[0],8,256))
    visc=np.reshape(visc[:,:,-1::-1],(visc0.shape[0],8*256))
    visc=visc/abs(visc)
    nt=visc.shape[0]
    nred=2
    ntr=nt/nred
    nt1=ntr*nred
    visc2=np.reshape(visc,(ntr,nred,256,8))
    times2=np.reshape(times,(ntr,nred)).mean(axis=1)
    viscm=visc2.mean(axis=3).mean(axis=1)
    plt.imshow(np.real(viscm.T),interpolation='nearest')

    islice=np.zeros((viscm.shape[0],viscm.shape[1]*nrfft),dtype=complex)
    islice[:,:viscm.shape[1]]=viscm
    cf1=np.fft.fft(islice,axis=1)
    cf1[:,27*nrfft:-27*nrfft]=0
    mloc=np.argmax(np.abs(cf1),axis=1)
    cf2=cf1-cf1

    cphase=np.zeros((viscm.shape[0]),dtype=complex)
    for i in range(cf1.shape[0]) :
            cf2[i,-mloc[i]]=np.conj(cf1[i,mloc[i]])
            cphase[i]=cf1[i,mloc[i]]
        
    cf3=np.fft.ifft(cf2,axis=1)
    tmp=cf3[:,:viscm.shape[1]]
    tmp=(tmp)/(abs(tmp)+1.e-20)
    vismcal=viscm*tmp

    viscal2=svd_model(vismcal)
    tmp2=tmp*viscal2/vismcal
    viscal2=np.reshape(visc2-visc2+tmp2[:,np.newaxis,:,np.newaxis],(nt1,64*32))

    lag=np.fft.fftshift(np.fft.fft(visc*viscal2,axis=1),axes=(1,))
    finecorr=abs(lag[:,1024])/lag[:,1024]
    viscal2=viscal2*finecorr[:,np.newaxis]

    finefreq=(visc*viscal2)[200:,:].mean(axis=0)
    finefreq=abs(finefreq)/finefreq
    viscal2=viscal2*finefreq
    viscal1=np.reshape(viscal2,(viscal2.shape[0],1024,2)).mean(axis=2)

    viscal1[:3*nt/4,:]=0

    u,s,w = np.linalg.svd(viscal1)

    vtemp=u[:,0]
    vtemp[:3*nt/4]=0
    nf=1000
    omega=4000.*(arange(nf)-nf/2.)/nf
    ft=(exp(-omega[:,np.newaxis]*times*(0.+1.j))*u[:,0]).mean(axis=1)
    fmax=np.argmax(abs(ft))
    tmodel=ft[fmax]*exp((0.+1.j)*times*omega[fmax])
    np.save('J1751model'+antenna+'ypol'+str(ipol)+'.npy',viscal1)
    np.save('J1751modelsvd'+antenna+'ypol'+str(ipol)+'.npy',(w[0,:],omega[fmax],ft[fmax]))

#np.save('sgramodel'+antenna+'ypol'+str(ipol)+'.npy',viscal1)

plt.imshow(np.real(lag),interpolation='nearest')



plt.imshow(np.real(visc*viscal2),interpolation='nearest')



lag=np.fft.fftshift(np.fft.fft(visc,axis=1),axes=(1,))
