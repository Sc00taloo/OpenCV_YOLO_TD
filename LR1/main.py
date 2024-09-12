import cv2
def opencv(input):
    img_color = cv2.imread(input, cv2.IMREAD_COLOR)
    img_gray = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    img_unchanged = cv2.imread(input, cv2.IMREAD_UNCHANGED)

    cv2.namedWindow('color', cv2.WINDOW_NORMAL)
    cv2.namedWindow('gray', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('unchanged', cv2.WINDOW_FULLSCREEN)

    cv2.imshow('color', img_color)
    cv2.imshow('gray', img_gray)
    cv2.imshow('unchanged', img_unchanged)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video(input):
    video = cv2.VideoCapture(input)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
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
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('original', img)
    cv2.imshow('HSV', img_hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def red_cross():
    video = cv2.VideoCapture(0)
    while True:
        ok, img = video.read()
        h, w, _ = img.shape
        center_x, center_y = w // 2, h // 2
        center_xx, center_yy = w // 2, h // 3
        cross_size = 70
        cv2.rectangle(img, (center_x - cross_size, center_y - 10), (center_x + cross_size, center_y + 10), (0, 0, 255), 2)
        cv2.rectangle(img, (center_xx - 10, center_yy), (center_xx + 10, center_yy + cross_size), (0, 0, 255), 2)
        cv2.rectangle(img, (center_xx - 10, center_y + 10), (center_xx + 10, center_y + 80), (0, 0, 255), 2)
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
        center_xx, center_yy = w // 2, h // 3
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
        cv2.rectangle(img, (center_x - cross_size, center_y - 10), (center_x + cross_size, center_y + 10), color, -1)
        cv2.rectangle(img, (center_xx - 10, center_yy), (center_xx + 10, center_yy + cross_size), color, -1)
        cv2.rectangle(img, (center_xx - 10, center_y + 10), (center_xx + 10, center_y + 80), color, -1)
        cv2.imshow('camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

def camera_iphone():
    video = cv2.VideoCapture(1)
    while True:
        ret, frame = video.read()
        cv2.imshow('camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #opencv('photo_2023-11-23_02-28-57.jpg')
    #video('toad.mp4')
    #convert_video('toad.mp4','toad.avi')
    #hsv_convert('IMG_6192.png')
    #red_cross()
    #recording_video()
    #full_cross()
    camera_iphone()
