from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as integrate


data = loadmat("Flyover_No_23/Flyover_No_23.mat")
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

### Voor p_prime kunnen wij onze data gebruiken. Hoeven we dus niet te berekenen via formule 1.2 ###

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
p_e0 = 10e-5 # Pa

for i, start in enumerate(time_points):
    end = start + T
    mask = (t >= start) & (t < end)
    integrals[i] = math.sqrt((integrate.trapezoid(sound_pressure[mask]**2, x=t[mask]))/T)
    OSPL[i] = 10 * math.log10(integrals[i]**2 / p_e0**2) # formula 5.20 from the reader

# plt.plot(t, sound_pressure, "orange", label="sound pressure")
# plt.plot(time_points, integrals, "green", label="effective pressure")
plt.plot(time_points, OSPL, "b", label="Overall sound pressure level")
plt.xlabel("Time [s]")
plt.ylabel("Sound pressure")
plt.title("Sound pressure over time")
plt.legend()
plt.show()

    #######################
    ### End of answer 5 ###
    #######################
    ###  Answer to q 6  ###
    #######################