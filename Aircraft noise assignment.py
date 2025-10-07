from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as integrate


data = loadmat("Flyover_No_23.mat")
# print(data.keys())

sound_pressure = np.squeeze(data["sound_pressure"])

# print(sound_pressure.shape)
# print(sound_pressure[:10]) 

    #######################
    ###  Answer to q 2  ###
    #######################

fs = 40000 #Hz
t = len(sound_pressure) / fs #seconds

print(f"Length of signal = {t} seconds")

    #######################
    ### End of answer 2 ###
    #######################
    ###  Answer to q 3  ###
    #######################

T = 0.1 #seconds
nr_samples = T * fs

print(f"Number of samples in {T} seconds = {nr_samples}")

    #######################
    ### End of answer 3 ###
    #######################
    ###  Answer to q 4  ###
    #######################

"""Voor p_prime kunnen wij onze data gebruiken. Hoeven we dus niet te berekenen via formule 1.2"""

t = np.arange(len(sound_pressure)) / fs     # Elke 1/40000 s een timestep voor 23.7s
time_points = np.arange(0, t[-1], T)        # Elke 1/10 s een timestep voor 23.7s
integrals = np.zeros_like(time_points)      # Elke 1/10 s een timestep met waarde 0 voor 23.7s

# print(len(t), time_points, integrals)

for i, start in enumerate(time_points):
    end = start + T
    mask = (t >= start) & (t < end)
    integrals[i] = math.sqrt((integrate.trapezoid(sound_pressure[mask]**2, x=t[mask]))/T)


# for chunk in np.arange(0, t, fs):
#     p_effective[chunk] = p_effective

plt.plot(t, sound_pressure, "orange", label="sound pressure")
plt.plot(time_points, integrals, "green", label="effective pressure")
plt.xlabel("Time [s]")
plt.ylabel("Sound pressure")
plt.title("Sound pressure over time")
plt.legend()
plt.show() # Dit plot zowel de grafiek voor opdracht 1 als voor opdracht 4

    #######################
    ### End of answer 4 ###
    #######################
    ###  Answer to q 5  ###
    #######################

OSPL = np.zeros_like(time_points) # Overall Sound Pressure Level empty array
p_e0 = 2e-5 # Pa

for i, start in enumerate(time_points):
    end = start + T
    mask = (t >= start) & (t < end)
    integrals[i] = math.sqrt((integrate.trapezoid(sound_pressure[mask]**2, x=t[mask]))/T)
    OSPL[i] = 10 * math.log10(integrals[i]**2 / p_e0**2) # formula 5.20 from the reader

# plt.plot(t, sound_pressure, "orange", label="sound pressure")
# plt.plot(time_points, integrals, "green", label="effective pressure")
plt.plot(time_points, OSPL, "b", label="Overall sound pressure level")
plt.xlabel("Time [s]")
plt.ylabel("Sound pressure [dB]")
plt.title("Sound pressure over time")
plt.legend()
plt.show()

    #######################
    ### End of answer 5 ###
    #######################
    ###  Answer to q 6  ###
    #######################

"""Om 6 te antwoorden moeten we eerst een Fourier transform gebruiken op onze data, 
om de PSD uit formule 5.13 te krijgen"""

import numpy as np
from scipy import signal, integrate
import matplotlib.pyplot as plt


chunk_length_s = 0.1    # seconds per chunk
chunk_samples = int(chunk_length_s * fs)

n_samples = len(sound_pressure)
duration = n_samples / fs

# create start times for non-overlapping chunks from 0 to final time
start_times = np.arange(0, duration, chunk_length_s)
n_chunks = len(start_times)

# time vector for plotting = center of each chunk
time_centers = start_times + chunk_length_s / 2.0

nfft = 2 ** int(np.ceil(np.log2(chunk_samples)))

freqs = np.fft.rfftfreq(nfft, d=1.0/fs)

PSL_matrix = np.zeros((n_chunks, len(freqs)))  # rows = time steps, cols = freqs

for i, start in enumerate(start_times):
    start_idx = int(np.round(start * fs))
    end_idx = start_idx + chunk_samples

    chunk = sound_pressure[start_idx:end_idx]
    if len(chunk) < chunk_samples:
        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')

    window = signal.windows.hann(len(chunk), sym=False)
    freqs_chunk, Pxx = signal.periodogram(chunk, fs=fs, window=window, nfft=nfft, scaling='density', detrend=False)

    PSL = 10.0 * np.log10(Pxx / (p_e0 ** 2) + 1e-30)  # small epsilon to avoid log(0)

    PSL_matrix[i, :] = PSL

plt.figure(figsize=(10, 5))
T, F = np.meshgrid(time_centers, freqs)  # shapes: freqs x times after transpose
pcm = plt.pcolormesh(T, F, PSL_matrix.T, shading='gouraud', cmap='jet', vmin=-5, vmax=70)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
cbar = plt.colorbar(pcm)
cbar.set_label('PSL (dB re p_ref^2)')
plt.title('Figure: Power Spectrum Level (time × frequency)')
plt.yscale('linear')        
plt.ylim(0, fs/2)          # show up to Nyquist
plt.tight_layout()
plt.show()
