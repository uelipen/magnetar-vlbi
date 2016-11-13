import astropy.io.fits as fits
import numpy as np
from pylab import *
import matplotlib.pylab as plt

nt=2173
ipol=0
for antenna in ['fd','la'] :
    (w,omega,ft)=np.load('J1751modelsvd'+antenna+'ypol'+str(ipol)+'.npy')
    times=np.load('time'+antenna+'y.npy')
    tmodel=ft*exp((0.+1.j)*times[:nt,np.newaxis]*omega)*w
    np.save('J1751model'+antenna+'ypol'+str(ipol)+'.npy',tmodel)
