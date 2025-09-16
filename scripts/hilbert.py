import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import hilbert

N = 1000
t = np.linspace(-8,8,N,1)

fs = len(t)/(t[-1] - t[1])

x = np.exp(-t**2)*np.cos(2*np.pi*2*t)

plt.plot(t,x)
plt.show()

anx = hilbert(x)
amplitude_envelope = np.abs(anx)
instantaneous_phase = np.unwrap(np.angle(anx))
instantaneous_frequency = np.diff(instantaneous_phase) / (2*np.pi) * fs

np.savetxt("data_hilbert_amp.csv", np.transpose([t,amplitude_envelope]), delimiter=",")
np.savetxt("data_hilbert_freq.csv", np.transpose([t[1:],instantaneous_frequency]), delimiter=",")

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex='all', tight_layout=True)
ax0.plot(t, amplitude_envelope)
ax1.plot(t[1:], instantaneous_frequency)

plt.show()
