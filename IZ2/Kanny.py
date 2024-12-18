import cv2
import numpy as np
import os

def scharr(gau):
    gaus_x = np.zeros_like(gau, dtype=np.float64)
    gaus_y = np.zeros_like(gau, dtype=np.float64)
    size = 3
    sobel_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    sobel_y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
    for i in range(size // 2, gau.shape[0] - size // 2):
        for j in range(size // 2, gau.shape[1] - size // 2):
            val_x = 0
            val_y = 0
            for k in range(-(size // 2), size // 2 + 1):
                for l in range(-(size // 2), size // 2 + 1):
                    val_x += gau[i + k, j + l] * sobel_x[k +(size // 2), l + (size // 2)]
                    val_y += gau[i + k, j + l] * sobel_y[k +(size // 2), l + (size // 2)]
            gaus_x[i, j] = val_x
            gaus_y[i,j] = val_y
    return gaus_x, gaus_y

def kann(img, parametrs, method):
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    size, sigma = parametrs[0], parametrs[1]
    gaus = cv2.GaussianBlur(image, ksize=(size, size), sigmaX=sigma, sigmaY=sigma)
    low_threshold = parametrs[2]
    high_threshold = parametrs[3]
    # test = cv2.Canny(image, low_threshold, high_threshold, L2gradient = True)
    # cv2.imshow('Filtered Image', cv2.resize(test, (960, 540)))
    if method == 'Yes':
        grad_x = cv2.Sobel(gaus, cv2.CV_64F, 1, 0)
        grad_y = cv2.Sobel(gaus, cv2.CV_64F, 0, 1)
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
        result = np.where(suppressed <= low_threshold, 0, np.where(suppressed >= high_threshold, 255, suppressed))
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if result[i, j] != 0 and result[i, j] != 255:
                    if np.any(result[i - 1:i + 1, j - 1:j + 1] == 255):
                        result[i, j] = 255
                    else:
                        result[i, j] = 0
        cv2.imshow('Sobel', cv2.resize(result, (960, 540)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    elif (method == 'Not'):
        grad = cv2.Laplacian(gaus, cv2.CV_64F, ksize=size)
        angle = cv2.phase(grad,grad, angleInDegrees=True)
        angle_new = angle % 180
        suppressed = np.zeros_like(grad, dtype=np.float64)
        rows, cols = grad.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if (0 <= angle_new[i, j] < 22.5) or (157.5 <= angle_new[i, j] <= 180):
                    neighbors = (grad[i, j - 1], grad[i, j + 1])
                elif 22.5 <= angle_new[i, j] < 67.5:
                    neighbors = (grad[i - 1, j + 1], grad[i + 1, j - 1])
                elif 67.5 <= angle_new[i, j] < 112.5:
                    neighbors = (grad[i - 1, j], grad[i + 1, j])
                else:
                    neighbors = (grad[i - 1, j - 1], grad[i + 1, j + 1])
                if grad[i, j] >= max(neighbors):
                    suppressed[i, j] = grad[i, j]
        result = np.where(suppressed <= low_threshold, 0, np.where(suppressed >= high_threshold, 255, suppressed))
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if result[i, j] != 0 and result[i, j] != 255:
                    if np.any(result[i - 1:i + 1, j - 1:j + 1] == 255):
                        result[i, j] = 255
                    else:
                        result[i, j] = 0
        cv2.imshow('Filtered Image', cv2.resize(result, (960, 540)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    elif (method == 'NotNot'):
        grad_x = cv2.Scharr(gaus, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(gaus, cv2.CV_64F, 0, 1)
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
        result = np.where(suppressed <= low_threshold, 0, np.where(suppressed >= high_threshold, 255, suppressed))
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if result[i, j] != 0 and result[i, j] != 255:
                    if np.any(result[i - 1:i + 1, j - 1:j + 1] == 255):
                        result[i, j] = 255
                    else:
                        result[i, j] = 0
        cv2.imshow('Filtered Image', cv2.resize(result, (960, 540)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    else:
        grad = cv2.Laplacian(gaus, cv2.CV_64F, ksize=size)
        angle = cv2.phase(grad,grad, angleInDegrees=True)
        grad_x = cv2.Sobel(gaus, cv2.CV_64F, 1, 0)
        grad_y = cv2.Sobel(gaus, cv2.CV_64F, 0, 1)

        grad_x_s = cv2.Scharr(gaus, cv2.CV_64F, 1, 0)
        grad_y_s = cv2.Scharr(gaus, cv2.CV_64F, 0, 1)
        dlina_gradient_s = cv2.magnitude(grad_x_s, grad_y_s)
        angle_gradient_s = cv2.phase(grad_x_s, grad_y_s, angleInDegrees=True)

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
        result = np.where(suppressed <= low_threshold, 0, np.where(suppressed >= high_threshold, 255, suppressed))
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if result[i, j] != 0 and result[i, j] != 255:
                    if np.any(result[i - 1:i + 1, j - 1:j + 1] == 255):
                        result[i, j] = 255
                    else:
                        result[i, j] = 0


        angle_new = angle % 180
        suppressed = np.zeros_like(grad, dtype=np.float64)
        rows, cols = grad.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if (0 <= angle_new[i, j] < 22.5) or (157.5 <= angle_new[i, j] <= 180):
                    neighbors = (grad[i, j - 1], grad[i, j + 1])
                elif 22.5 <= angle_new[i, j] < 67.5:
                    neighbors = (grad[i - 1, j + 1], grad[i + 1, j - 1])
                elif 67.5 <= angle_new[i, j] < 112.5:
                    neighbors = (grad[i - 1, j], grad[i + 1, j])
                else:
                    neighbors = (grad[i - 1, j - 1], grad[i + 1, j + 1])
                if grad[i, j] >= max(neighbors):
                    suppressed[i, j] = grad[i, j]
        result_not = np.where(suppressed <= low_threshold, 0, np.where(suppressed >= high_threshold, 255, suppressed))
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if result_not[i, j] != 0 and result_not[i, j] != 255:
                    if np.any(result_not[i - 1:i + 1, j - 1:j + 1] == 255):
                        result_not[i, j] = 255
                    else:
                        result_not[i, j] = 0

        angle_new = angle_gradient_s % 180
        suppressed = np.zeros_like(dlina_gradient_s, dtype=np.float64)
        rows, cols = dlina_gradient_s.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if (0 <= angle_new[i, j] < 22.5) or (157.5 <= angle_new[i, j] <= 180):
                    neighbors = (dlina_gradient_s[i, j - 1], dlina_gradient_s[i, j + 1])
                elif 22.5 <= angle_new[i, j] < 67.5:
                    neighbors = (dlina_gradient_s[i - 1, j + 1], dlina_gradient_s[i + 1, j - 1])
                elif 67.5 <= angle_new[i, j] < 112.5:
                    neighbors = (dlina_gradient_s[i - 1, j], dlina_gradient_s[i + 1, j])
                else:
                    neighbors = (dlina_gradient_s[i - 1, j - 1], dlina_gradient_s[i + 1, j + 1])
                if grad[i, j] >= max(neighbors):
                    suppressed[i, j] = dlina_gradient_s[i, j]
        result_not_not = np.where(suppressed <= low_threshold, 0, np.where(suppressed >= high_threshold, 255, suppressed))
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if result_not_not[i, j] != 0 and result_not_not[i, j] != 255:
                    if np.any(result_not_not[i - 1:i + 1, j - 1:j + 1] == 255):
                        result_not_not[i, j] = 255
                    else:
                        result_not_not[i, j] = 0
        cv2.imshow('Sobel', cv2.resize(result, (960, 540)))
        cv2.imshow('Not Sobel', cv2.resize(result_not, (960, 540)))
        cv2.imshow('Not Not Sobel', cv2.resize(result_not_not, (960, 540)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def tracking(input, method):
    folder_path = "Mans/"
    image_files = sorted(os.listdir(folder_path))
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        kann(image_path, input, method)

def algo(input):
    return 0