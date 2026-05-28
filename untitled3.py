import numpy as np
import matplotlib.pyplot as plt

ampl = 3
f0 = 10
min_limit_x = 0
max_limit_x = 1
N = 174

fd = N / max_limit_x
t = np.linspace(min_limit_x, max_limit_x, N)

mas_psp4 = [2, 6, 11, 1, 5, 8, 6, 9, 5, 5]

casual_signal = []
signal_prp4 = []

for i in t:
    tmp = int(i // 0.1)
    freq = mas_psp4[tmp]
    casual_signal.append(ampl * np.sin(2 * np.pi * f0 * i))
    signal_prp4.append(ampl * np.sin(2 * np.pi * f0 * freq * i))

plt.figure(1)
plt.plot(t, signal_prp4, 'orange', label='Сигнал с ПСПЧ')
plt.plot(t, casual_signal, label='Обычный ВЧ-сигнал')
plt.title('Временные реализации сигнала с ПСПЧ и ВЧ-сигнала')
plt.grid(True)
plt.legend()

M = 2 * N
f_range = np.zeros(M * 2)
for smp_ref in range(-int(M), int(M)):
    f_range[smp_ref] = (smp_ref / M / 2) * fd

fft_result_casual = np.fft.fft(casual_signal, M * 2) / N
fft_result_prp4 = np.fft.fft(signal_prp4, M * 2) / N

mod_fftshift_result_casual = abs(fft_result_casual)
mod_fftshift_result_prp4 = abs(fft_result_prp4)

plt.figure(2)
plt.plot(f_range, mod_fftshift_result_casual, label='Спектр ВЧ-сигнала')
plt.plot(f_range, mod_fftshift_result_prp4, 'orange', label='Спектр сигнала с ПСПЧ')
plt.title('Спектры сигнала с ПСПЧ и ВЧ-сигнала')
plt.grid(True)
plt.legend()

plt.show()