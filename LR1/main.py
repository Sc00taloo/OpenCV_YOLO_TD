import math

import cv2
def opencv(input):
    # Чтение цветного изображения
    img_color = cv2.imread(input, cv2.IMREAD_COLOR)
    # Чтение изображения в градациях серого
    img_gray = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    # Чтение изображения без изменений
    img_unchanged = cv2.imread(input, cv2.IMREAD_UNCHANGED)
    # Создание окна для цветного изображения
    cv2.namedWindow('color', cv2.WINDOW_NORMAL)
    # Создание окна для серого изображения с автоматическим размером
    cv2.namedWindow('gray', cv2.WINDOW_AUTOSIZE)
    # Создание полноэкранного окна для неизмененного изображения
    cv2.namedWindow('unchanged', cv2.WINDOW_FULLSCREEN)

    cv2.imshow('color', img_color)
    cv2.imshow('gray', img_gray)
    cv2.imshow('unchanged', img_unchanged)
    # Ожидание нажатия клавиши
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video(input):
    # Производим захват видео
    video = cv2.VideoCapture(input)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        # Чтение кадра из видео
        ok, img = video.read()
        if not ok:
            break
        # Изменение размера видео на норму
        cv2.namedWindow('norma', cv2.WINDOW_NORMAL)
        # Из BRGA (blue, green, red, alpha) в RGBA (red, green, blue, alpha)
        norma_blue = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        # Изменение размера видео
        resized = cv2.resize(img, (w // 2, h // 2))
        # Серый цвет видео
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Преобразование в HSV видео
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        cv2.imshow('img',img)
        cv2.imshow('norma', norma_blue)
        cv2.imshow('resized', resized)
        cv2.imshow('gray', gray)
        cv2.imshow('HSV', hsv)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

def convert_video(input, output):
    video = cv2.VideoCapture(input)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Определение кодека и создание объекта VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  #XVID кодек
    video_write = cv2.VideoWriter(output, fourcc, 25, (w, h))
    while True:
        ok, img = video.read()
        if not ok:
            break
        video_write.write(img)
    video.release()
    video_write.release()
    cv2.destroyAllWindows()

def hsv_convert(input):
    img = cv2.imread(input)
    # Преобразование изображения в формат HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('original', img)
    cv2.imshow('HSV', img_hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def red_cross():
    video = cv2.VideoCapture(0)
    while True:
        ok, img = video.read()
        # Возвращает кортеж в формате (высота, ширина, количество каналов)
        h, w, _ = img.shape
        center_x, center_y = w // 2, h // 2
        cross_size = 70

        top_left1 = (center_x+140, center_y+80)
        bottom_right1 = (center_x-140, center_y-80)
        cv2.rectangle(img, top_left1, bottom_right1, (255, 255, 255), -1)

        top_left = (center_x+140, center_y+40)
        bottom_right = (center_x-140, center_y+70)
        cv2.rectangle(img, top_left, bottom_right, (255,0,0), -1)

        top_left2 = (center_x+140, center_y-40)
        bottom_right2 = (center_x-140, center_y-70)
        cv2.rectangle(img, top_left2, bottom_right2, (255, 0, 0), -1)

        T = int(0.06875*center_x)
        D = center_y - 8 * T
        n = 6
        agle = -90
        vertical = []
        for i in range(n):
            x = int(center_x + D//2 * math.cos(math.radians(agle + (i * 360) / n)))
            y = int(center_y + D // 2 * math.sin(math.radians(agle + (i * 360) / n)))
            vertical.append((x,y))
        for i in range(n):
            cv2.line(img, vertical[i], vertical[(i+2) % n], (255,0,0), 2)
       #cv2.circle(img, (center_x, center_y), cross_size, (0, 0, 255), 2)

        # start_point = (center_x-65, center_y-25)  # Начальная точка (x, y)
        # end_point = (center_x+65, center_y-25)  # Конечная точка (x, y)
        # cv2.line(img, start_point, end_point,(0, 0, 255), 2)
        #
        # start_point2 = (center_x+65, center_y-25)  # Начальная точка (x, y)
        # end_point2 = (center_x-45, center_y+55)  # Конечная точка (x, y)
        # cv2.line(img, start_point2, end_point2, (0, 0, 255), 2)
        #
        # start_point3 = (center_x-45, center_y+55)  # Начальная точка (x, y)
        # end_point3 = (center_x, center_y-70)  # Конечная точка (x, y)
        # cv2.line(img, start_point3, end_point3, (0, 0, 255), 2)
        #
        # start_point4 = (center_x, center_y-70)  # Начальная точка (x, y)
        # end_point4 = (center_x+45, center_y+55)  # Конечная точка (x, y)
        # cv2.line(img, start_point4, end_point4, (0, 0, 255), 2)
        #
        # start_point5 = (center_x+45, center_y+55)  # Начальная точка (x, y)
        # end_point5 = (center_x-65, center_y-25)  # Конечная точка (x, y)
        # cv2.line(img, start_point5, end_point5, (0, 0, 255), 2)

        cv2.imshow('camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

def recording_video():
    video = cv2.VideoCapture(0)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output.mp4v", fourcc, 25, (w,h))
    while True:
        ok, img = video.read()
        cv2.imshow('img',img)
        video_writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

def full_cross():
    video = cv2.VideoCapture(0)
    while True:
        ok, img = video.read()
        h, w, _ = img.shape
        center_x, center_y = w // 2, h // 2
        # Будет содержать значение цвета пикселя в координатах x и y
        center_pixel = img[center_y, center_x]

        # Красный
        if center_pixel[2] > center_pixel[1] and center_pixel[2] > center_pixel[0]:
            color = (0, 0, 255)
        # Зеленый
        elif center_pixel[1] > center_pixel[2] and center_pixel[1] > center_pixel[0]:
            color = (0, 255, 0)
        # Синий
        else:
            color = (255, 0, 0)

        cross_size = 70
        cv2.circle(img, (center_x, center_y), cross_size, (0, 0, 255), 2)

        start_point = (center_x-65, center_y-25)  # Начальная точка (x, y)
        end_point = (center_x+65, center_y-25)  # Конечная точка (x, y)
        cv2.line(img, start_point, end_point,color, 2)

        start_point2 = (center_x+65, center_y-25)  # Начальная точка (x, y)
        end_point2 = (center_x-45, center_y+55)  # Конечная точка (x, y)
        cv2.line(img, start_point2, end_point2, color, 2)

        start_point3 = (center_x-45, center_y+55)  # Начальная точка (x, y)
        end_point3 = (center_x, center_y-70)  # Конечная точка (x, y)
        cv2.line(img, start_point3, end_point3, color, 2)

        start_point4 = (center_x, center_y-70)  # Начальная точка (x, y)
        end_point4 = (center_x+45, center_y+55)  # Конечная точка (x, y)
        cv2.line(img, start_point4, end_point4, color, 2)

        start_point5 = (center_x+45, center_y+55)  # Начальная точка (x, y)
        end_point5 = (center_x-65, center_y-25)  # Конечная точка (x, y)
        cv2.line(img, start_point5, end_point5, color, 2)
        cv2.imshow('camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

def camera_iphone():
    video1 = cv2.VideoCapture("https://192.168.43.63:8080/video")
    #video = cv2.VideoCapture(1)
    while True:
        ret, frame = video1.read()
        cv2.imshow('camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video1.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #opencv('photo_2023-11-23_02-28-57.jpg')
    #video('toad.mp4')
    #convert_video('toad.mp4','toad.avi')
    #hsv_convert('IMG_6192.png')
    red_cross()
    #recording_video()
    #full_cross()
    #camera_iphone()
