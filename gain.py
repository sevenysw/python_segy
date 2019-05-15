import numpy as np
import math
from scipy.signal.windows import triang
from scipy.signal import convolve2d as conv2

def gain(data,dt,option1,parameters,option2):
    '''
    GAIN: Gain a group of traces.
    
      gain(d,dt,option1,parameters,option2);
    
      IN   d(nt,nx):   traces
           dt:         sampling interval
           option1 = 'time' parameters = [a,b],  gain = t.^a . * exp(-bt)
                   = 'agc' parameters = [agc_gate], length of the agc gate in secs
           option2 = 0  No normalization
                   = 1  Normalize each trace by amplitude
                   = 2  Normalize each trace by rms value
    
      OUT  dout(nt,nx): traces after application of gain function
    '''

    nt,nx = data.shape

    dout = np.zeros(data.shape)
    if option1 == 'time':
        a = parameters[0]
        b = parameters[1]
        t = [x*dt for x in range(nt)]
        tgain = [(x**a)*math.exp(x*b) for x in t]

        for k in range(nx):
            dout[:,k] = data[:,k]*tgain


    elif option1 == 'agc':
        L = parameters/dt+1
        L = np.floor(L/2)
        h = triang(2*L+1)
        shaped_h  = h.reshape(len(h),1)

        for k in range(nx):
            aux = data[:,k]
            e = aux**2
            shaped_e = e.reshape(len(e),1)
            
            rms = np.sqrt(conv2(shaped_e,shaped_h,"same"))
            epsi = 1e-10*max(rms)
            op = rms/(rms**2+epsi)
            op = op.reshape(len(op),)

            dout[:,k] = data[:,k]*op

    #Normalize by amplitude 
    if option2==1:
        for k in range(nx):
            aux =  dout[:,k]
            amax = max(abs(aux))
            dout[:,k] = dout[:,k]/amax 

    #Normalize by rms 
    if option2==2:
        for k in range(nx):
            aux = dout[:,k]
            amax = np.sqrt(sum(aux**2)/nt)
            dout[:,k] = dout[:,k]/amax


    return dout       