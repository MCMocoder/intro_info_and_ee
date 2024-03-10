import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import scipy.signal as sig
from scipy import fftpack
from scipy import integrate
import librosa as lr
import pyaudio as pa

def fird(x):
    firb = sig.firwin(100, 20, fs=sr, window='hamming')
    fira = 1
    return sig.lfilter(firb, fira, x)


def play(y):
    raw=np.array([int(s*32768) for s in y], "<i2").tobytes()
    p = pa.PyAudio()
    stream = p.open(22050,1,pa.paInt16,output=True) # 频率，通道，int16
    stream.write(raw)


# Digital Part
sr = 2e2
timestep = 1/sr
cf = 10
cf2 = 15
slen=2000
dlen=20
window=sig.windows.hann(slen)


tn = np.linspace(0, slen, slen, endpoint = False)
car = np.array([np.cos(n * 2 * np.pi*timestep * cf) for n in tn])
car2 = np.array([np.cos(n * 2 * np.pi*timestep * cf2) for n in tn])
carl = np.array([np.sin(n * 2 * np.pi*timestep * cf) for n in tn])
rd=np.random.randint(0,2,size=dlen)
rdc=np.repeat(rd,slen//dlen)
dcspec = fftpack.fftshift(fftpack.fft(window*rdc)/slen*2)

# 0C
spec = fftpack.fftshift(fftpack.fft(window*rdc))
plt.plot(fftpack.fftshift(fftpack.fftfreq(len(rdc),timestep)),20*np.log(np.abs(spec)/slen*2))
plt.show()

# N0C
rds = np.zeros(2*len(rd),dtype=int)
rds[::2] = rd
rds[1::2] = 0
rdsc=np.repeat(rds,slen//len(rds))
plt.plot(rdsc)
plt.plot(rdc)
plt.show()
spec = fftpack.fftshift(fftpack.fft(window*rdsc))
plt.plot(fftpack.fftshift(fftpack.fftfreq(len(rdc),timestep)),20*np.log(np.abs(spec)/slen*2))
plt.show()

# ASK
mod=(rdc*0.8+0.2)*car
plt.plot(tn*timestep,mod)
plt.show()
spec = fftpack.fftshift(fftpack.fft(window*mod))
plt.plot(fftpack.fftshift(fftpack.fftfreq(len(mod),timestep)),20*np.log(np.abs(spec)/slen*2))
plt.show()

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

# Analog Part
sr = 2e6
timestep = 1/sr
cf = 2e5
df1 = 30000
df2 = 50000
df3 = 20000
slen=2000

window=sig.windows.hann(slen)
tn = np.linspace(0, slen, slen, endpoint = False)
car = np.array([np.cos(n * 2 * np.pi*timestep * cf) for n in tn])
carl = np.array([np.sin(n * 2 * np.pi*timestep * cf) for n in tn])
audio1 = np.array([np.cos(n * 2 * np.pi*timestep * df1) for n in tn])
audio2 = np.array([np.cos(n * 2 * np.pi*timestep * df2) for n in tn])
audio3 = np.array([np.cos(n * 2 * np.pi*timestep * df3) for n in tn])

audio = audio1+audio2+audio3

# AM
mod = car*(1+0.5*audio[0:slen])
plt.plot(tn*timestep,mod)
plt.show()
spec = fftpack.fftshift(fftpack.fft(window*mod))
plt.plot(fftpack.fftshift(fftpack.fftfreq(len(mod),timestep)),20*np.log(np.abs(spec)/slen*2))
plt.show()

# DSB
mod=audio*car
plt.plot(tn*timestep,mod)
plt.show()
spec = fftpack.fftshift(fftpack.fft(window*mod))
plt.plot(fftpack.fftshift(fftpack.fftfreq(len(mod),timestep)),20*np.log(np.abs(spec)/slen*2))
plt.show()

# SSB
rr=np.imag(sig.hilbert(audio))
mod=audio*car-rr*carl
plt.plot(tn*timestep,np.real(mod))
plt.show()
spec = fftpack.fftshift(fftpack.fft(window*mod))
plt.plot(fftpack.fftshift(fftpack.fftfreq(len(mod),timestep)),20*np.log(np.abs(spec)/slen*2))
plt.show()

# FM
iaudio = np.cumsum(audio)
mod = np.array([np.cos(n * 2 * np.pi*timestep * cf+0.05*iaudio[int(n)]) for n in tn])
plt.plot((tn*timestep)[0:300],mod[0:300])
plt.show()
spec = fftpack.fftshift(fftpack.fft(window*mod))
plt.plot(fftpack.fftshift(fftpack.fftfreq(len(mod),timestep)),20*np.log(np.abs(spec)/slen*2))
plt.show()
