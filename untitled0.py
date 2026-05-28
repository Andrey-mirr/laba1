import numpy as np
import matplotlib.pyplot as plt

N = 174
mean = 3
sigma = 1.6
amplitudes = [1, 2, 1, 4, 5, 8, 9]
frequencies = [16, 3, 0, 2, 8, 9, 11]

t = np.linspace(0, 2 * np.pi, N)

signal_sum = np.zeros(N)
for i in range(len(amplitudes)):
    signal_sum += amplitudes[i] * np.sin(frequencies[i] * t)

noise = np.random.normal(mean, sigma, N)

additive_mix = signal_sum + noise

raw_spectrum = np.fft.fft(additive_mix)
ampl_spectrum = abs(raw_spectrum)
shift_ampl_spectrum = np.fft.fftshift(ampl_spectrum / (N/2))

reconstructed_signal = np.fft.ifft(raw_spectrum)
reconstructed_signal_real = np.real(reconstructed_signal)

signal_difference = reconstructed_signal_real - additive_mix

plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(noise, 'r')
plt.title('1. Нормальный шум')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(signal_sum, 'b')
plt.title('2. Полезный сигнал (сумма 7 синусоид)')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(additive_mix, 'g')
plt.title('3. Аддитивная смесь (сигнал + шум)')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(shift_ampl_spectrum, 'm')
plt.title('4. Спектр аддитивной смеси')
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(reconstructed_signal_real, 'c')
plt.title('5. Сигнал после обратного Фурье')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(signal_difference, 'k')
plt.title('6. Погрешность восстановления')
plt.grid(True)

plt.tight_layout()
plt.show()