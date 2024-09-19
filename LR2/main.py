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
        cv2.imshow('HSV', hsv_record)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    record.release()
    cv2.destroyAllWindows()

def filter_inRange():
    record = cv2.VideoCapture(0)
    # Определеяем диапазон красного цвета в HSV
    min_red = np.array([0, 100, 100]) # минимальные значения оттенка, насыщенности и яркости
    max_red = np.array([255, 255, 255]) # максимальные значения оттенка, насыщенности и яркости
    while True:
        ok, frame = record.read()
        hsv_record = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if not ok:
            break
        # Маска - бинарное изображение, где пиксели, соответствующие заданному диапазону цвета, имеют значение 255 (белый), а остальные пиксели имеют значение 0 (черный).
        mask = cv2.inRange(hsv_record, min_red, max_red)
        # Применение маски на кадр(побитовая операция И между изображениями)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("HSV", hsv_record)
        cv2.imshow("HSV with red", res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    record.release()
    cv2.destroyAllWindows()

def morphological_transformation():
    record = cv2.VideoCapture(0)
    # Определяем структурный элемент для морфологических операций (определяет размер и форму области)
    kernel = np.ones((5, 5), np.uint8)
    min_red = np.array([0, 100, 100])
    max_red = np.array([255, 255, 255])
    while True:
        ok, frame = record.read()
        if not ok:
            break
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, min_red, max_red)
        # Операция открытие - позволяет удалить шумы и мелкие объекты на изображении(удаление нежелательных пикселей или деталей)
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
    min_red = np.array([0, 100, 100])
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
        cv2.imshow('Original Video', frame)
        cv2.imshow('Filtered Red', hsv_record)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    record.release()
    cv2.destroyAllWindows()

def black_rectangle():
    record = cv2.VideoCapture(0)
    min_red = np.array([0, 100, 100])
    max_red = np.array([255, 255, 255])
    while True:
        ok, frame = record.read()
        if not ok:
            break
        hsv_record = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_record, min_red, max_red)
        moments = cv2.moments(mask, True)
        # Поиск первого момента по оси y
        dM01 = moments['m01']
        # Поиск первого момента по оси x
        dM10 = moments['m10']
        area = moments['m00']
        # Проверка, достаточно ли велика площадь для рисования прямоугольника
        if area > 5000:
            # Вычисление координат центра масс
            x = int(dM10 / area)
            y = int(dM01 / area)
            width = height = int(np.sqrt(area))
            top_left = (x - width, y - height)
            bottom_right = (x + width, y + height)
            cv2.rectangle(frame, top_left, bottom_right, (0,0,0), 4)
        cv2.putText(frame, f'Area: {int(area)}', (10, 30), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
        cv2.imshow('HSV_frame', hsv_record)
        cv2.imshow('Result_frame', frame)
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