import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import scipy.signal as sig
from scipy import fftpack
from scipy import integrate


sr = 2e2
timestep = 1/sr
cf = 10
cf2 = 15
slen=2000
dlen=20
window=sig.windows.hann(slen)

def fir(x):
    firb = sig.firwin(100, 20, fs=sr, window='hamming')
    fira = 1
    return sig.lfilter(firb, fira, x)

tn = np.linspace(0, slen, slen, endpoint = False)
car = np.array([np.cos(n * 2 * np.pi*timestep * cf) for n in tn])
car2 = np.array([np.cos(n * 2 * np.pi*timestep * cf2) for n in tn])
carl = np.array([np.sin(n * 2 * np.pi*timestep * cf) for n in tn])
rd=np.random.randint(0,2,size=dlen)
rdc=np.repeat(rd,slen//dlen)
dcspec = fftpack.fftshift(fftpack.fft(window*rdc)/slen*2)
# ASK
#mod=(rdc*0.8+0.2)*car
#plt.plot(tn*timestep,mod)
#plt.show()
#spec = fftpack.fftshift(fftpack.fft(window*mod))
#plt.plot(fftpack.fftshift(fftpack.fftfreq(len(mod),timestep)),20*np.log(np.abs(spec)/slen*2))
#plt.show()
#
## DSB
#mod=rdc*car
#plt.plot(tn*timestep,mod)
#plt.show()
#spec = fftpack.fftshift(fftpack.fft(window*mod))
#plt.plot(fftpack.fftshift(fftpack.fftfreq(len(mod),timestep)),20*np.log(np.abs(spec)/slen*2))
#plt.show()
#
## SSB
#rdch=sig.hilbert(rdc)
#mod=rdc*car-rdch*carl
#plt.plot(tn*timestep,np.real(mod))
#plt.show()
#spec = fftpack.fftshift(fftpack.fft(window*mod))
#plt.plot(fftpack.fftshift(fftpack.fftfreq(len(mod),timestep)),20*np.log(np.abs(spec)/slen*2))
#plt.show()

# 2FSK
mod=car*rdc+car2*(1-rdc)
plt.plot(tn*timestep,np.real(mod))
plt.show()
spec = fftpack.fftshift(fftpack.fft(window*mod))
plt.plot(fftpack.fftshift(fftpack.fftfreq(len(mod),timestep)),20*np.log(np.abs(spec)/slen*2))
plt.show()

# 2PSK
mod=car*rdc-car*(1-rdc)
plt.plot(tn*timestep,np.real(mod))
plt.show()
spec = fftpack.fftshift(fftpack.fft(window*mod))
plt.plot(fftpack.fftshift(fftpack.fftfreq(len(mod),timestep)),20*np.log(np.abs(spec)/slen*2))
plt.show()

# AM
