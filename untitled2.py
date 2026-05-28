import numpy as np
import matplotlib.pyplot as plt

N = 174
mean = 3
sigma = 1.6
amplitudes = [1, 2, 1, 4, 5, 8, 9]
frequencies = [16, 3, 0, 2, 8, 9, 11]

min_limit_t = 0
max_limit_t = 1
t = np.linspace(min_limit_t, max_limit_t, N)

fd = N / max_limit_t

signal_sum = np.zeros(N)
for i in range(len(amplitudes)):
    signal_sum += amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t)

noise = np.random.normal(mean, sigma, N)

additive_mix = signal_sum + noise

plt.figure()
plt.plot(t, additive_mix)
plt.title('Аддитивная смесь (сигнал + шум)')
plt.grid(True)

M = 4 * N
f_range = np.zeros(M)
for smp_ref in range(-int(M/2), int(M/2)):
    f_range[smp_ref] = (smp_ref / M) * fd

fft_result = np.fft.fft(additive_mix, M) / N
mod_fftshift_result = abs(fft_result)

plt.figure()
plt.plot(f_range, 2 * mod_fftshift_result)
plt.title('Спектр аддитивной смеси')
plt.grid(True)

freq_rez = 8
polosa = 10

min_a4h_freq = freq_rez - int(polosa / 2)
max_a4h_freq = freq_rez + int(polosa / 2)

print("=" * 50)
print("ПАРАМЕТРЫ ФИЛЬТРА")
print("=" * 50)
print(f"Резонансная частота (freq_rez): {freq_rez} Гц")
print(f"Полоса пропускания (polosa): {polosa} Гц")
print(f"Нижняя граница полосы: {min_a4h_freq} Гц")
print(f"Верхняя граница полосы: {max_a4h_freq} Гц")
print("=" * 50)

a4h = np.zeros(M)
for smp_ref in range(-int(M/2), int(M/2)):
    if (f_range[smp_ref] > min_a4h_freq) and (f_range[smp_ref] < max_a4h_freq):
        a4h[smp_ref] = 1
    if (f_range[smp_ref] < -min_a4h_freq) and (f_range[smp_ref] > -max_a4h_freq):
        a4h[smp_ref] = 1

plt.figure()
plt.plot(f_range, a4h, f_range, mod_fftshift_result / max(mod_fftshift_result))
plt.title('АЧХ фильтра и спектр сигнала')
plt.grid(True)

filtered_spectrum = a4h * fft_result
filtered_signal = N * np.fft.ifft(filtered_spectrum)
invert_fft_mas_len = int(len(filtered_signal) * N / M)

plt.figure()
plt.plot(t, filtered_signal[0:invert_fft_mas_len])
plt.title('Отфильтрованная аддитивная смесь')
plt.grid(True)

plt.figure()
plt.plot(f_range, a4h * mod_fftshift_result)
plt.title('Отфильтрованный спектр')
plt.grid(True)

plt.show()