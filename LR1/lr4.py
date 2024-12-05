import cv2
import numpy as np

def gray_gaus_conclusion(input):
    image = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    imge = cv2.resize(image, (960, 540))
    sizes_and_sigmas = [(5, 3)]
    for size, sigma in sizes_and_sigmas:
        gaus = cv2.GaussianBlur(image, ksize=(size, size), sigmaX=sigma, sigmaY=sigma)
        img = cv2.resize(gaus, (960, 540))
        cv2.imshow(f'Gaus {size}', img)
    cv2.imshow('Original', imge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def two_matrix(input):
    image = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    sizes_and_sigmas = [(5, 3)]
    for size, sigma in sizes_and_sigmas:
        gaus = cv2.GaussianBlur(image, ksize=(size, size), sigmaX=sigma, sigmaY=sigma)
        # Зададим матрицы оператора Собеля, создаем матрицы для значений частных производных, длины градиента и угла градиента
        grad_x = cv2.Sobel(gaus, cv2.CV_64F, 1, 0, ksize=5)
        #cv2.imshow('x', grad_x)
        grad_y = cv2.Sobel(gaus, cv2.CV_64F, 0, 1, ksize=5)
        #cv2.imshow('y', grad_y)
        dlina_gradient = cv2.magnitude(grad_x, grad_y)
        angle_gradient = cv2.phase(grad_x, grad_y, angleInDegrees=True)
        print("Матрица длин градиентов:")
        print(dlina_gradient)
        print("\nМатрица углов градиентов:")
        print(angle_gradient)
        dlina = cv2.resize(dlina_gradient, (960, 540))
        angle = cv2.resize(angle_gradient, (960, 540))
        cv2.imshow('Dlina gradient', cv2.convertScaleAbs(dlina))
        cv2.imshow('Angle gradient', angle / 360)
    #cv2.imshow('Original', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def not_max(input):
    image = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    sizes_and_sigmas = [(5, 3)]
    for size, sigma in sizes_and_sigmas:
        gaus = cv2.GaussianBlur(image, ksize=(size, size), sigmaX=sigma, sigmaY=sigma)
        grad_x = cv2.Sobel(gaus, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gaus, cv2.CV_64F, 0, 1, ksize=5)
        dlina_gradient = cv2.magnitude(grad_x, grad_y)
        angle_gradient = cv2.phase(grad_x, grad_y, angleInDegrees=True)
        dlina = cv2.resize(dlina_gradient, (960, 540))
        angle = cv2.resize(angle_gradient, (960, 540))

        angle_new = angle_gradient % 180
        # Инициализация матрицы результата
        suppressed = np.zeros_like(dlina_gradient, dtype=np.float64)
        rows, cols = dlina_gradient.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # Определение направления градиента
                if (0 <= angle_new[i, j] < 22.5) or (157.5 <= angle_new[i, j] <= 180):
                    neighbors = (dlina_gradient[i, j - 1], dlina_gradient[i, j + 1])
                elif 22.5 <= angle_new[i, j] < 67.5:
                    neighbors = (dlina_gradient[i - 1, j + 1], dlina_gradient[i + 1, j - 1])
                elif 67.5 <= angle_new[i, j] < 112.5:
                    neighbors = (dlina_gradient[i - 1, j], dlina_gradient[i + 1, j])
                else:
                    neighbors = (dlina_gradient[i - 1, j - 1], dlina_gradient[i + 1, j + 1])
                # Подавление пикселей, которые не являются локальными максимумами
                if dlina_gradient[i, j] >= max(neighbors):
                    suppressed[i, j] = dlina_gradient[i, j]
        non_max = cv2.resize(suppressed, (960, 540))

        cv2.imshow('Dlina gradient', cv2.convertScaleAbs(dlina))
        cv2.imshow('Angle gradient', angle / 360)
        cv2.imshow('Non_max', cv2.convertScaleAbs(non_max))
    #cv2.imshow('Original', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def nothing(*arg):
    pass

def double_filter(input):
    image = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    size, sigma = 5, 4
    gaus = cv2.GaussianBlur(image, ksize=(size, size), sigmaX=sigma, sigmaY=sigma)
    grad_x = cv2.Sobel(gaus, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gaus, cv2.CV_64F, 0, 1, ksize=5)
    dlina_gradient = cv2.magnitude(grad_x, grad_y)
    angle_gradient = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    angle_new = angle_gradient % 180
    suppressed = np.zeros_like(dlina_gradient, dtype=np.float64)
    rows, cols = dlina_gradient.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if (0 <= angle_new[i, j] < 22.5) or (157.5 <= angle_new[i, j] <= 180):
                neighbors = (dlina_gradient[i, j - 1], dlina_gradient[i, j + 1])
            elif 22.5 <= angle_new[i, j] < 67.5:
                neighbors = (dlina_gradient[i - 1, j + 1], dlina_gradient[i + 1, j - 1])
            elif 67.5 <= angle_new[i, j] < 112.5:
                neighbors = (dlina_gradient[i - 1, j], dlina_gradient[i + 1, j])
            else:
                neighbors = (dlina_gradient[i - 1, j - 1], dlina_gradient[i + 1, j + 1])
            if dlina_gradient[i, j] >= max(neighbors):
                suppressed[i, j] = dlina_gradient[i, j]

    cv2.namedWindow("settings")
    cv2.createTrackbar('low_threshold', 'settings', 50, 255, nothing)
    cv2.createTrackbar('high_threshold', 'settings', 100, 255, nothing)
    while True:
        low_threshold = cv2.getTrackbarPos('low_threshold', 'settings')
        high_threshold = cv2.getTrackbarPos('high_threshold', 'settings')
        result = np.where(suppressed <= low_threshold, 0, np.where(suppressed >= high_threshold, 255, suppressed))
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if result[i, j] != 0 and result[i, j] != 255:
                    if np.any(result[i - 1:i + 1, j - 1:j + 1] == 255):
                        result[i, j] = 255
                    else:
                        result[i, j] = 0
        cv2.imshow('Filtered Image', cv2.resize(result, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #gray_gaus_conclusion("eo6v6k01xua51.png")
    #two_matrix("eo6v6k01xua51.png")
    #not_max("eo6v6k01xua51.png")
    double_filter("eo6v6k01xua51.png")