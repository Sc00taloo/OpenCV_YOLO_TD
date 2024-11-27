import numpy as np

def fourier(image):
    M, N = image.shape
    result = np.zeros((M, N), dtype=complex)
    # Координата по частотам по вертикали
    for u in range(M):
        # Координата по частотам по горизонтали
        for v in range(N):
            summation = 0
            for m in range(M):
                for n in range(N):
                    summation += image[m, n] * np.exp(-2j * np.pi * ((u * m / M) + (v * n / N)))
            result[u, v] = summation
    return result

def dft(image):
    M, N = image.shape
    result = np.zeros((M, N), dtype=complex)
    for x in range(M):
        for y in range(N):
            summation = 0
            for u in range(M):
                for v in range(N):
                    summation += image[u, v] * np.exp(2j * np.pi * ((u * x / M) + (v * y / N)))
            result[x, y] = summation/(M*N)
    return result


image = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=float)

my = dft(image)
notMy = np.fft.ifft2(image)
# Выводим результаты
print("Результат не встроенного:")
print(np.round(my, 2))
print("Результат встроенного:")
print(np.round(notMy, 2))

my = fourier(image)
notMy = np.fft.fft2(image)
# Выводим результаты
print("\nРезультат не встроенного:")
print(np.round(my, 2))
print("Результат встроенного:")
print(np.round(notMy, 2))