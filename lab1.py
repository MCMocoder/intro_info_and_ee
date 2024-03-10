import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import scipy.signal as ssig
from scipy import fftpack

sr = 2<<20
timestep = 1/(2<<20)

# Nyquist Freq:524288Hz
def iir(t,x):
    b, a = ssig.butter(4, [25e3 / (2 << 19)])
    freq, response = ssig.freqz(b,a, fs=2 << 20)
    amplitude_response = 20 * np.log10(abs(response))
    plt.figure()
    plt.plot(freq / np.pi, amplitude_response)
    plt.title('IIR Frequency Response')
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Amplitude(dB)')
    plt.grid(True)
    plt.show()
    sig=ssig.filtfilt(b,a,x)
    fs = fftpack.fft(sig*window) / len(sig) * 2
    fs = fftpack.fftshift(fs)
    plt.plot(fftpack.fftshift(fftpack.fftfreq(len(sig), timestep))[len(sig) // 2:len(sig) // 2 + len(sig) // 30],
             np.abs(fs)[len(sig) // 2:len(sig) // 2 + len(sig) // 30])
    plt.plot(freq[:len(freq) // 15], np.abs(response)[:len(response) // 15])
    plt.show()



def fir(x):
    firb = ssig.firwin(100, 25e3, fs=2<<20, window='hamming')
    fira = 1
    freq, response = ssig.freqz(firb,fs=2<<20)
    amplitude_response = 20 * np.log10(abs(response))
    plt.figure()
    plt.plot(freq/np.pi, amplitude_response)
    plt.title('FIR Frequency Response')
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Amplitude(dB)')
    plt.grid(True)
    plt.show()
    sig = ssig.lfilter(firb, fira, x)
    fs = fftpack.fft(sig*window) / len(sig) * 2
    fs = fftpack.fftshift(fs)
    plt.plot(fftpack.fftshift(fftpack.fftfreq(len(sig), timestep))[len(sig) // 2:len(sig) // 2+len(sig) // 30],
             np.abs(fs)[len(sig) // 2:len(sig) // 2+len(sig) // 30])
    plt.plot(freq[:len(freq)//15],np.abs(response)[:len(response)//15])
    plt.show()

def mix(x):
    sig = x**2+x
    fs = fftpack.fft(sig*window) / len(sig) * 2
    fs = fftpack.fftshift(fs)
    plt.plot(fftpack.fftshift(fftpack.fftfreq(len(sig), timestep))[len(sig) // 2:len(sig) // 2 + len(sig) // 30],
             np.abs(fs)[len(sig) // 2:len(sig) // 2 + len(sig) // 30])
    plt.show()


t = np.linspace(0.0, 8192, 8192, endpoint = False)
sig1 = np.array([np.cos(f * 2 * np.pi*timestep*1e3 * 15.195) for f in t]) # 128kHz
sig2 = np.array([np.cos(f * 2 * np.pi*timestep*1e3 * 23.352) for f in t]) # 256kHz
sig3 = np.array([np.cos(f * 2 * np.pi*timestep*1e3 * 34.235) for f in t]) # 384kHz

sign = sig1 + sig2 + sig3

window = ssig.windows.hamming(len(sign))*2
#window = 1

sig =sign*window

# Time domain
plt.plot((t * timestep)[0:1280], sig[0:1280])
plt.show()

# FFT
fs = fftpack.fft(sig) / len(sig) * 2
fs = fftpack.fftshift(fs)
plt.subplot(211)
plt.xlabel("Frequency(Hz)")
plt.ylabel("Magnitude")
plt.plot(fftpack.fftshift(fftpack.fftfreq(len(sig), timestep))[len(sig) // 2:len(sig)//2+len(sig)//30], np.abs(fs)[len(sig) // 2:len(sig)//2+len(sig)//30])
plt.subplot(212)
plt.xlabel("Frequency(Hz)")
plt.ylabel("Phase")
plt.plot(fftpack.fftshift(fftpack.fftfreq(len(sig), timestep))[len(sig) // 2:len(sig)//2+len(sig)//30], np.angle(fs)[len(sig) // 2:len(sig)//2+len(sig)//30]*180/np.pi)
plt.show()

# FIR Filter(3-stage)
fir(sign)

# IIR Filter(3-stage,butterworth)
iir(t,sign)

# Nonlinear system(mixer,detector)
mix(sig)