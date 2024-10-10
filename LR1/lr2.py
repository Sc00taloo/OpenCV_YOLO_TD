import cv2
import numpy as np
def HSV_recording():
    record = cv2.VideoCapture(0)
    while True:
        ok, frame = record.read()
        if not ok:
            break
        # Преобразование изображения в формат HSV
        hsv_record = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('HSV frame', hsv_record)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    record.release()
    cv2.destroyAllWindows()

def filter_inRange():
    record = cv2.VideoCapture(0)
    # Определеяем диапазон красного цвета в HSV
    min_red = np.array([0, 100, 0]) # минимальные значения оттенка, насыщенности и яркости
    max_red = np.array([255, 255, 255]) # максимальные значения оттенка, насыщенности и яркости
    while True:
        ok, frame = record.read()
        hsv_record = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if not ok:
            break
        # Маска - бинарное изображение, где пиксели, соответствующие заданному диапазону цвета, имеют значение 255 (белый), а остальные пиксели имеют значение 0 (черный)
        mask = cv2.inRange(hsv_record, min_red, max_red)
        # Применение маски к оригинальному изображению, оставляя только красные области (побитовая операция И между изображениями)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("HSV frame", hsv_record)
        cv2.imshow("HSV with red frame", res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    record.release()
    cv2.destroyAllWindows()

def morphological_transformation():
    record = cv2.VideoCapture(0)
    # Определяем структурный элемент для морфологических операций (определяет размер и форму области)
    # Ядро - это матрица, которая применяется к каждому пикселю изображения для вычисления нового
    # значения пикселя на основе значений соседних пикселей.
    # np.uint8 - это тип данных, который будет использоваться для хранения элементов массива.
    # uint8 есть целые числа без знака, занимающие 8 бит (от 0 до 255).
    kernel = np.ones((5, 5), np.uint8)
    min_red = np.array([0, 100, 0])
    max_red = np.array([255, 255, 255])
    while True:
        ok, frame = record.read()
        if not ok:
            break
        hsv_record = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_record, min_red, max_red)
        # Операция открытие - позволяет удалить шумы и мелкие объекты на изображении
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Операция закрытие - позволяет заполнить маленькие пробелы и разрывы в объектах на изображении
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("HSV with red", mask)
        cv2.imshow('Opening', opening)
        cv2.imshow('Closing', closing)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    record.release()
    cv2.destroyAllWindows()

def get_moments():
    record = cv2.VideoCapture(0)
    min_red = np.array([100, 100, 100])
    max_red = np.array([255, 255, 255])
    while True:
        ok, frame = record.read()
        if not ok:
            break
        hsv_record = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_record, min_red, max_red)
        # Вычисление момента на основе маски
        moments = cv2.moments(mask, True)
        # Поиск момента первого порядка (площадь)
        area = moments['m00']
        # Отображение площади на изображении
        cv2.putText(frame, f'Area: {int(area)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Отображение оригинального фрейма и результат морфологических операций
        cv2.imshow('Original frame', frame)
        cv2.imshow('Filtered frame', mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    record.release()
    cv2.destroyAllWindows()

def nothing(*arg):
        pass
def black_rectangle():
    record = cv2.VideoCapture(0)
    min_red = np.array([114, 102, 77])
    max_red = np.array([255, 255, 255])

    # cv2.namedWindow("settings")  # создаем окно настроек
    # cv2.createTrackbar('h1', 'settings', 0, 255, nothing)
    # cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
    # cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
    # cv2.createTrackbar('h2', 'settings', 255, 255, nothing)
    # cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
    # cv2.createTrackbar('v2', 'settings', 255, 255, nothing)

    while True:
        ok, frame = record.read()
        if not ok:
            break

        # # считываем значения бегунков
        # h1 = cv2.getTrackbarPos('h1', 'settings')
        # s1 = cv2.getTrackbarPos('s1', 'settings')
        # v1 = cv2.getTrackbarPos('v1', 'settings')
        # h2 = cv2.getTrackbarPos('h2', 'settings')
        # s2 = cv2.getTrackbarPos('s2', 'settings')
        # v2 = cv2.getTrackbarPos('v2', 'settings')

        # min_red = np.array([h1, s1, v1])
        # max_red = np.array([h2, s2, v2])

        hsv_record = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_record, min_red, max_red)
        moments = cv2.moments(mask, True)
        # Поиск первого момента по оси y
        dM01 = moments['m01']
        # Поиск первого момента по оси x
        dM10 = moments['m10']
        area = moments['m00']
        # Проверка, достаточно ли велика площадь для рисования прямоугольника
        if area > 5500:
            # Вычисление координат центра масс
            x = int(dM10 / area)
            y = int(dM01 / area)
            w = h = int(np.sqrt(area))
            center_x, center_y = x + w, y + h
            cross_size = 360
            top_left = (x - w, y - h)
            bottom_right = (x + w, y + h)
            cv2.circle(frame, (x, y), cross_size+h, (0, 0, 0), 800)
            start_point = (x, y+h+70)  # Начальная точка (x, y)
            end_point = (x, y-h-70)  # Конечная точка (x, y)
            cv2.line(frame, start_point, end_point, (0, 0, 0), 4)

            start_point1 = (x+h+70, y)  # Начальная точка (x, y)
            end_point1 = (x-h-70, y)  # Конечная точка (x, y)
            cv2.line(frame, start_point1, end_point1, (0, 0, 0), 4)

            #cv2.rectangle(frame, top_left, bottom_right, (0,0,0), 4)
        cv2.putText(frame, f'Area: {int(area)}', (10, 30), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
        cv2.imshow('HSV frame', mask)
        cv2.imshow('Original frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    record.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #HSV_recording()
    #filter_inRange()
    #morphological_transformation()
    #get_moments()
    black_rectangle()