from copy import copy
import cv2
import numpy as np

class BoundingBox:
    # x: Координата центра по x
    # y: Координата центра по y
    # w: Ширина прямоугольника
    # h: Высота прямоугольника
    # object_x: Точная координата центра по x
    # object_y: Точная координата центра по y
    def __init__(self, x: int, y: int, w: int, h: int, object_x: int = None, object_y: int = None):
        self.x = x
        self.y = y
        if w % 2 == 0:
            self.w = w
        else:
            self.w = w + 1
        if h % 2 == 0:
            self.h = h
        else:
            self.h = h + 1
        if object_x is not None:
            self.object_x = object_x
        else:
            self.object_x = x
        if object_y is not None:
            self.object_y = object_y
        else:
            self.object_y = y

    # height: Высота кадра
    # width: Ширина кадра
    # target_bbox_h: Основная высота окна поиска
    # target_bbox_w: Основная ширина окна поиска
    # Получаем True, если прямоугольник был обрезан, иначе False
    def clip(self, height: int, width: int, target_bbox_h: int, target_bbox_w: int):
        # Ограничиваем центр по координатам X и Y, чтобы он оставался в пределах кадра
        x = max(0, min(self.x, width))
        y = max(0, min(self.y, height))

        # Корректируем размеры окна поиска, если они выходят за границы кадра
        w, h = target_bbox_w, target_bbox_h
        if self.x < target_bbox_w // 2:
            w = 2 * self.x
        if self.y < target_bbox_h // 2:
            h = 2 * self.y
        if target_bbox_w > 2 * (width - self.x):
            w = 2 * (width - self.x)
        if target_bbox_h > 2 * (height - self.y):
            h = 2 * (height - self.y)

        # Проверяем на изменяемость параметров
        clipped = x != self.x or y != self.y or w != self.w or h != self.h
        # Обновляем параметры
        self.x, self.y, self.w, self.h = x, y, w, h
        return clipped

    # Выщитывание границы прямоугольника
    def left(self):
        return self.x - self.w // 2
    def top(self):
        return self.y - self.h // 2
    def right(self):
        return self.x + self.w // 2
    def bottom(self):
        return self.y + self.h // 2


class MosseResult:
    # bbox: прямоугольник
    # target_response: гауссовский отклик
    # A: Накопленная матрица числителя фильтра
    # B: Накопленная матрица знаменателя фильтра
    # frame_tru: Изображение, прошедшее предобработку
    # Инициализируем параметры для работы фильтра MOSSE
    def __init__(self,frame, bbox: BoundingBox, target_response, A, B, frame_tru):
        self.frame = frame
        self.bbox = bbox
        self.target_response = target_response
        # Вычисляем сам фильтр как отношение A и B
        self.filter = A / B
        # Применяем фильтр к изображению
        G = self.filter * np.fft.fft2(frame_tru)
        # Нормализуем фильтр и отклик в пространственной области
        self.filter = np.abs(normalize(np.fft.ifft2(self.filter)))
        self.output = np.abs(normalize(np.fft.ifft2(G)))

def gaus(height: int, width: int, center_x: int, center_y: int, sigma: float = 10.0):
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))  # Сетка координат
    gaus = (1 / (np.pi * 2 * (sigma ** 2))) * (np.exp(-((xs-center_x)**2 + (ys-center_y)**2)/(2*sigma**2)))
    gaus = normalize(gaus)
    return gaus

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

# Логарифмическое преобразование изображения для усиления слабых сигналов
def log_transform(image):
    return np.log(image + 1)

def normalize_image(image):
    return (image - image.mean()) / (image.std() + 1e-5)

# Применение окна Ханнинга для уменьшения краевых эффектов
def hanning_window(image):
    height, width = image.shape
    mask_col, mask_row = np.meshgrid(np.hanning(width), np.hanning(height))
    window = mask_col * mask_row
    return image * window

# Предобработка изображения
def preprocess(image):
    image = log_transform(image)
    image = normalize_image(image)
    image = hanning_window(image)
    return image

# bbox: прямоугольник
# rotation: Угол поворота
# scale: Масштабирование
# translation: Сдвиг пикселей
# Применяем случайное вращение, масштабирование и сдвиг к изображению и области объекта для аугментации данных
def random_affine_transform(image, bbox: BoundingBox, rotation: float = 180 / 16, scale: float = 0.05, translation: int = 4):
    new_bbox = copy(bbox)
    # Генерируем угол поворота
    angle = np.random.uniform(-rotation, rotation)
    matrix = cv2.getRotationMatrix2D((new_bbox.x, new_bbox.y), angle, 1.0)
    image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    # Применение масштабирования
    scale = np.random.uniform(1 - scale, 1 + scale)
    image = cv2.resize(image, None, fx=scale, fy=scale)
    # Применение сдвига
    trans_x = np.random.randint(-translation, translation)
    trans_y = np.random.randint(-translation, translation)
    new_bbox.object_x += trans_x
    new_bbox.object_y += trans_y

    return image, new_bbox


class Mosse:
    def __init__(self, sigma: float = 10, learning_rate: float = 0.125):
        self.sigma = sigma
        # Коэффициент обучения
        self.learning_rate = learning_rate
        # Для отслеживания первого кадра
        self.first_frame = True
        # Для отслеживания выхода объекта за границы
        self.clipped = False

    # Инициализация и предобучение фильтра на первом кадре
    def init(self,frame,bbox: BoundingBox,pretrain_iters: int = 128):
        self.bbox = bbox
        # Высота окна поиска
        self.search_win_h = self.bbox.h
        # Ширина окна поиска
        self.search_win_w = self.bbox.w
        results = []

        # Преобразуем кадр в оттенки серого
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Генерируем гауссовский отклик с центром в области объекта
        response = gaus(*frame.shape, bbox.x, bbox.y)
        # Обрезаем отклик и изображение до размеров области объекта
        # g: гауссовский отклик
        # f: Изображение объекта
        g = response[self.bbox.top() : self.bbox.bottom(), self.bbox.left() : self.bbox.right()]
        f = frame[self.bbox.top() : self.bbox.bottom(), self.bbox.left() : self.bbox.right()]

        # Преобразуем отклик и изображение в частотную область
        # np.fft.fft2: вычисляет двумерное дискретное преобразование Фурье
        # G: дискретное преобразование Фурье для гаусоввого отлика
        G = np.fft.fft2(g)
        # Предобработка изображения
        f = preprocess(f)
        # F: дискретное преобразование Фурье для изображения
        F = np.fft.fft2(f)
        # Инициализируем матрицы фильтра A и B
        # Числитель фильтра
        self.A_i = G * np.conj(F)
        # Знаменатель фильтра
        self.B_i = F * np.conj(F)
        # Сохраняем результат
        results.append(MosseResult(frame, self.bbox, response, self.A_i, self.B_i, f))

        # Аугментируем данные. Повторяем обучение на случайных трансформациях
        for _ in range(pretrain_iters):
            # Генерируем случайно трансформированный кадр и область объекта
            trans_frame, bbox = random_affine_transform(frame, self.bbox)

            # Создаем гауссовский отклик для нового положения объекта
            response = gaus(*trans_frame.shape, bbox.object_x, bbox.object_y)
            # Обрезаем отклик и изображение под область объекта
            g = response[bbox.top() : bbox.bottom(), bbox.left() : bbox.right()]
            f = trans_frame[bbox.top() : bbox.bottom(), bbox.left() : bbox.right()]

            # Преобразуем отклик и изображение в частотную область
            G = np.fft.fft2(g)
            # Предобработка изображения
            f = preprocess(f)
            F = np.fft.fft2(f)
            # Обновляем матрицы фильтра
            self.A_i += G * np.conj(F)
            self.B_i += F * np.conj(F)
            # Сохраняем результат
            results.append(MosseResult(trans_frame, self.bbox, response, self.A_i, self.B_i, f))

        # Применяем скорость обучения
        self.A_i *= self.learning_rate
        self.B_i *= self.learning_rate

        # Вычисляем фильтр H_i - MOSSE в частотной области
        self.H_i = self.A_i / self.B_i

        # Сохраняем результат
        results.append(MosseResult(frame, self.bbox, response, self.A_i, self.B_i, f))
        return results

    # Обновление фильтра и положения объекта на новом кадре
    def update(self, frame):
        # Кадр в серый
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Вырезаем область из текущего кадра
        f = frame[self.bbox.top() : self.bbox.bottom(), self.bbox.left() : self.bbox.right()]
        # Предобработка области
        f = preprocess(f)

        # Если размеры области изменились, корректируем их
        if f.shape != (self.search_win_h, self.search_win_w):
            f = cv2.resize(f, (self.search_win_w, self.search_win_h))
            # Вышли за границы
            self.clipped = False

        # Применяем фильтр к изображению
        # Применяем корреляцию, где f - обработанное изображение области
        G = self.H_i * np.fft.fft2(f)
        g = normalize(np.fft.ifft2(G))

        # Проверка наличия NaN в g
        if g.size == 0 or np.isnan(g).any() or not np.isfinite(g).all():
            return self.bbox

        # Поиск максимума в g
        max_pos = np.where(g == g.max())
        if len(max_pos[0]) == 0 or len(max_pos[1]) == 0:
            return self.bbox

        # Обновление координат bbox
        self.bbox.x = int(max_pos[1].mean() + self.bbox.left())
        self.bbox.y = int(max_pos[0].mean() + self.bbox.top())

        # Проверка выхода за границы
        self.clipped = self.bbox.clip(*frame.shape, self.search_win_h, self.search_win_w)
        if not (0 <= self.bbox.x < frame.shape[1] and 0 <= self.bbox.y < frame.shape[0]):
            return self.bbox

        # Обновляем фильтр с учетом нового положения объекта
        f = frame[self.bbox.top() : self.bbox.bottom(), self.bbox.left() : self.bbox.right()]
        # Предобработка области
        f = preprocess(f)

        # Если размеры области изменились, корректируем их
        if f.shape != (self.search_win_h, self.search_win_w):
            if f is None or f.size == 0:
                return None
            f = cv2.resize(f, (self.search_win_w, self.search_win_h))
        # Применяем фильтр к изображению
        G = self.H_i * np.fft.fft2(f)

        # Обновляем A и B с учетом нового кадра
        self.A_i = (self.learning_rate * (G * np.conj(np.fft.fft2(f)))+ (1 - self.learning_rate) * self.A_i)
        self.B_i = (self.learning_rate * (np.fft.fft2(f) * np.conj(np.fft.fft2(f)))+ (1 - self.learning_rate) * self.B_i)
        # Пересчитываем фильтр H_i
        self.H_i = self.A_i / self.B_i

        return self.bbox
