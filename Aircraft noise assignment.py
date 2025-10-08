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

from scipy import signal

nfft = 2 ** int(np.ceil(np.log2(nr_samples)))   # rond het aantal samples in 0.1 s omhoog van 4000 naar 4096

# f = rfftfreq(nr_samples, 1/fs) # samples van 0 hz tot Nyquist frequency
freqs = np.fft.rfftfreq(nfft, d=1.0/fs)         # dit bepaalt de brackets waarin de frequenties worden opgedeeld

PSL_matrix = np.zeros((len(time_points), len(freqs)))  # rows = time steps, cols = freqs, lege matrix

for i, start in enumerate(time_points):
    start_idx = int(np.round(start * fs))
    end_idx = start_idx + int(nr_samples)
    # print(start_idx, end_idx)
    chunk = sound_pressure[start_idx:end_idx] # metingen pakken van tijdstip start tot end
    if len(chunk) < nr_samples:
        # zero-pad to chunk_samples
        chunk = np.pad(chunk, (0, int(nr_samples) - len(chunk)), mode='constant')

    # compute PSD using periodogram with 'density' scaling (units Pa^2/Hz)
    freqs_chunk, Pxx = signal.periodogram(chunk, fs=fs, window='boxcar', nfft=nfft, scaling='density', detrend=False)

    PSL = 10.0 * np.log10(Pxx / (p_e0 ** 2) + 1e-30)  # Pxx (PSD) converteren naar PSL, epsilon = 1e-30

    PSL_matrix[i, :] = PSL


time_centers = time_points + T / 2.0

plt.figure(figsize=(10, 5))
T, F = np.meshgrid(time_centers, freqs)
pcm = plt.pcolormesh(T, F, PSL_matrix.T, shading='gouraud', cmap='jet', vmin=-10, vmax=70)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
cbar = plt.colorbar(pcm)
cbar.set_label('PSL (dB re p_ref^2)')
plt.title('Figure: Power Spectrum Level (time x frequency)')
plt.yscale('linear')        
plt.ylim(0, fs/2)          # show up to Nyquist
plt.tight_layout()
plt.show()