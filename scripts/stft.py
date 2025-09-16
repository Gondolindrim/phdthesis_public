import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
rng = np.random.default_rng()

fs = 10e3
N = 1e6
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)

fstart = 2e3
fend = 4e3

mod = 0#500*np.cos(2*np.pi*0.25*time)
carrier = amp * np.concatenate((np.sin(2*np.pi*fstart*time + mod),np.sin(2*np.pi*fend*time + mod)))

#noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
#noise *= np.exp(-time/5)
#x = carrier + noise


x = carrier

#plt.plot(np.concatenate((time,time + time[-1])),carrier)
#plt.show()

scaling = 1
f, t, Zxx = signal.stft(x, 1000, nperseg=10000*scaling)

print(Zxx.shape)

savedata = []
maxabsZ = np.abs(Zxx)/np.max(np.abs(Zxx)) # Normalizing
for tindex in range(len(t)):
	for findex in range(len(f)):
		savedata.append([t[tindex],f[findex],maxabsZ[findex,tindex]])

np.savetxt("data_stft_{}.csv".format(scaling), savedata, delimiter=",")


scaled_t = t*scaling
plt.pcolormesh(scaled_t, f/scaling, np.abs(Zxx), vmin=0, vmax=amp,cmap='binary')
#plt.xlim([9.95,10.05])

#plt.pcolormesh([k if 9.95 < k and k < 1.05 else 0 for k in scaled_t], f/scaling, np.abs(Zxx), vmin=0, vmax=amp,cmap='binary')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
