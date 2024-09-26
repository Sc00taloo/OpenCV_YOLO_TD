import cv2
import numpy as np
def gauss(x, y, sigma, a, b):
    # Вычисляет нормализационный множитель так, чтобы интеграл от гауссовой функции по всей плоскости был = 1.
    # Это обеспечивает, что сумма всех значений гауссовой функции будет равна 1.
    m1 = 1 / (np.pi * 2 * (sigma ** 2))  # 2pi*sigma^2
    # Вычисляет экспоненциальное значение с отрицательным аргументом.
    # (x-a) ** 2 и (y-b) ** 2 - квадраты расстояний от точки (x, y) до центра гауссовой функции (a, b).
    m2 = np.exp(-((x - a) ** 2 + (y - b) ** 2) / (2 * sigma ** 2))
    return m1 * m2

def gaus_1_2():
    standard_deviation = 1
    kernel = np.ones((3, 3))
    a = b = (3 + 1) // 2
    # Построение матрицы свёртки
    for i in range(3):
        for j in range(3):
            kernel[i, j] = gauss(i, j, standard_deviation, a, b)
    print("Матрица Гаусса 3x3:\n", (kernel))
    kernel = np.ones((5, 5))
    a = b = (5 + 1) // 2
    for i in range(5):
        for j in range(5):
            kernel[i, j] = gauss(i, j, standard_deviation, a, b)
    print("\nМатрица Гаусса 5x5:\n", kernel)
    kernel = np.ones((7, 7))
    a = b = (7 + 1) // 2
    for i in range(7):
        for j in range(7):
            kernel[i, j] = gauss(i, j, standard_deviation, a, b)
    print("\nМатрица Гаусса 7x7:\n", kernel)

def normalize_kernel(kernel):
    return kernel / np.sum(kernel)
def gaus_normalize():
    standard_deviation = 1
    kernel = np.ones((3, 3))
    a = b = (3 + 1) // 2
    # Построение матрицы свёртки
    for i in range(3):
        for j in range(3):
            kernel[i, j] = gauss(i, j, standard_deviation, a, b)
    normalized_kernel_3x3 = normalize_kernel(kernel)
    print("Матрица Гаусса 3x3:\n", normalized_kernel_3x3)
    kernel = np.ones((5, 5))
    a = b = (5 + 1) // 2
    for i in range(5):
        for j in range(5):
            kernel[i, j] = gauss(i, j, standard_deviation, a, b)
    normalized_kernel_5x5 = normalize_kernel(kernel)
    print("\nМатрица Гаусса 5x5:\n", normalized_kernel_5x5)
    kernel = np.ones((7, 7))
    a = b = (7 + 1) // 2
    for i in range(7):
        for j in range(7):
            kernel[i, j] = gauss(i, j, standard_deviation, a, b)
    normalized_kernel_7x7 = normalize_kernel(kernel)
    print("\nМатрица Гаусса 7x7:\n", normalized_kernel_7x7)

def apply_gaussian_filter(image, kernel):
    size = len(kernel)
    imageBlur = image.copy()
    # Начальные координаты для итераций по пикселям
    x_start = size // 2
    y_start = size // 2
    for i in range(x_start, imageBlur.shape[0] - x_start):
        for j in range(y_start, imageBlur.shape[1] - y_start):
            # Операция свёртки - каждый пиксель умножается на соответствующий элемент ядра свертки, а затем все произведения суммируются
            val = 0
            for k in range(-(size // 2), size // 2 + 1):
                for l in range(-(size // 2), size // 2 + 1):
                    val += image[i + k, j + l] * kernel[k +(size // 2), l + (size // 2)]
            imageBlur[i, j] = val
    return imageBlur
def gaus_filter():
    standard_deviation = 2
    kernel = np.ones((5, 5))
    a = b = (5 + 1) // 2
    # Построение матрицы свёртки
    for i in range(5):
        for j in range(5):
            kernel[i, j] = gauss(i, j, standard_deviation, a, b)
    normalized_kernel = normalize_kernel(kernel)
    image = cv2.imread("photo_2023-11-23_02-28-57.jpg", cv2.IMREAD_COLOR)
    # Применение фильтра Гаусса
    filtered_image = apply_gaussian_filter(image, normalized_kernel)
    cv2.imshow('Gaus', filtered_image)
    cv2.imshow('Original', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gaus_filter_tru():
    image = cv2.imread("photo_2023-11-23_02-28-57.jpg", cv2.IMREAD_COLOR)
    sizes_and_sigmas = [(3, 1), (5, 2), (7, 3), (9, 4)]
    for size, sigma in sizes_and_sigmas:
        standard_deviation = sigma
        kernel = np.ones((size, size))
        a = b = (size + 1) // 2
        # Построение матрицы свёртки
        for i in range(size):
            for j in range(size):
                kernel[i, j] = gauss(i, j, standard_deviation, a, b)
        normalized_kernel = normalize_kernel(kernel)
        filtered_image = apply_gaussian_filter(image, normalized_kernel)
        cv2.imshow(f'Gaus {size}', filtered_image)
    cv2.imshow('Original', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def opencv_gaus():
    image = cv2.imread("photo_2023-11-23_02-28-57.jpg", cv2.IMREAD_COLOR)
    sizes_and_sigmas = [(3, 1), (9, 4)]
    for size, sigma in sizes_and_sigmas:
        standard_deviation = sigma
        kernel = np.ones((size, size))
        a = b = (size + 1) // 2
        # Построение матрицы свёртки
        for i in range(size):
            for j in range(size):
                kernel[i, j] = gauss(i, j, standard_deviation, a, b)
        normalized_kernel = normalize_kernel(kernel)
        filtered_image = apply_gaussian_filter(image, normalized_kernel)
        cv2.imshow(f'My Gaus {size}', filtered_image)
    for size, sigma in sizes_and_sigmas:
        blurred7 = cv2.GaussianBlur(image, ksize=(size, size), sigmaX=sigma, sigmaY=sigma)
        cv2.imshow(f'CV Gaus {size}', blurred7)
    cv2.imshow('Original', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #gaus_1_2()
    #gaus_normalize()
    #gaus_filter()
    #gaus_filter_tru()
    opencv_gaus()