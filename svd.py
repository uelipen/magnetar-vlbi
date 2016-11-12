import numpy as np

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
   s[1:] = 0.0
   S = np.zeros([len(u), len(w)], np.complex128)
   S[:len(s), :len(s)] = np.diag(s)


   model = np.dot(np.dot(u, S), w)

   if phase_only==True:
       return arr * np.exp(-1j * np.angle(model))
   else:
       return arr / (model)


