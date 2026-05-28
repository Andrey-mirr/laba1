import numpy as np
import matplotlib.pyplot as plt

ampl = 3
f0 = 10
min_limit_x = 0
max_limit_x = 1
N = 174

fd = N / max_limit_x
t = np.linspace(min_limit_x, max_limit_x, N)

period = 0.05
mas_len = int(max_limit_x / period)
mas_manip = [-1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1]

freq_manip_sig = []
phas_manip_sig = []
ampl_manip_sig = []
casual_signal = []
input_sig = []

f1 = 40
fd_manip = 20

for i in t:
    tmp = int(i // period)
    casual_signal.append(ampl * np.sin(2 * np.pi * f1 * i))
    input_sig.append(mas_manip[tmp])
    
    ampl_manip_sig.append((ampl + mas_manip[tmp]) * np.sin(2 * np.pi * f1 * i))
    
    phas_manip_sig.append(ampl * np.sin(2 * np.pi * f1 * i + np.pi * (mas_manip[tmp] / 2 - 0.5)))
    
    freq_manip_sig.append(ampl * np.sin(2 * np.pi * (f1 + fd_manip * mas_manip[tmp]) * i))

plt.figure(1)
plt.plot(t, input_sig, label='Информационный сигнал')
plt.plot(t, ampl_manip_sig, label='АМн сигнал')
plt.title('Амплитудная манипуляция (АМн)')
plt.grid(True)
plt.legend()

plt.figure(2)
plt.plot(t, input_sig, label='Информационный сигнал')
plt.plot(t, phas_manip_sig, label='ФМн сигнал')
plt.title('Фазовая манипуляция (ФМн)')
plt.grid(True)
plt.legend()

plt.figure(3)
plt.plot(t, input_sig, label='Информационный сигнал')
plt.plot(t, freq_manip_sig, label='ЧМн сигнал')
plt.title('Частотная манипуляция (ЧМн)')
plt.grid(True)
plt.legend()

M = 2 * N
f_range = np.zeros(M * 2)
for smp_ref in range(-int(M), int(M)):
    f_range[smp_ref] = (smp_ref / M / 2) * fd

fft_result_casual = np.fft.fft(casual_signal, M * 2) / N
fft_result_ampl_manip = np.fft.fft(ampl_manip_sig, M * 2) / N
fft_result_phas_manip = np.fft.fft(phas_manip_sig, M * 2) / N
fft_result_freq_manip = np.fft.fft(freq_manip_sig, M * 2) / N

mod_fftshift_result_casual = abs(fft_result_casual)
mod_fftshift_result_ampl_manip = abs(fft_result_ampl_manip)
mod_fftshift_result_phas_manip = abs(fft_result_phas_manip)
mod_fftshift_result_freq_manip = abs(fft_result_freq_manip)

plt.figure(4)
plt.plot(f_range, mod_fftshift_result_ampl_manip, label='Спектр АМн')
plt.plot(f_range, mod_fftshift_result_casual, label='Спектр несущей')
plt.title('Спектр Амплитудной манипуляции (АМн)')
plt.grid(True)
plt.legend()

plt.figure(5)
plt.plot(f_range, mod_fftshift_result_phas_manip, label='Спектр ФМн')
plt.plot(f_range, mod_fftshift_result_casual, label='Спектр несущей')
plt.title('Спектр Фазовой манипуляции (ФМн)')
plt.grid(True)
plt.legend()

plt.figure(6)
plt.plot(f_range, mod_fftshift_result_freq_manip, label='Спектр ЧМн')
plt.plot(f_range, mod_fftshift_result_casual, label='Спектр несущей')
plt.title('Спектр Частотной манипуляции (ЧМн)')
plt.grid(True)
plt.legend()

plt.show()

