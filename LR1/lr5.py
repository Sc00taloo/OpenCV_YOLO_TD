import cv2
def lr5(kernel, standard_deviation, delta_tresh, min_area, i):
    video = cv2.VideoCapture('1.mp4', cv2.CAP_ANY)
    ret, frame = video.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (kernel, kernel), standard_deviation)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('newVideo' + str(i) + '.mp4', fourcc, 25, (w, h))
    while True:
        old_img = img.copy()
        ok, frame = video.read()
        if not ok:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (kernel, kernel), standard_deviation)

        frame_diff = cv2.absdiff(img, old_img)
        thresh = cv2.threshold(frame_diff, delta_tresh, 255, cv2.THRESH_BINARY)[1]
        contors, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contr in contors:
            area = cv2.contourArea(contr)
            if area < min_area:
                continue
            video_writer.write(frame)
    video_writer.release()

if __name__ == '__main__':
    parametrs = [(3, 4, 60, 20, 1),(3, 4, 40, 5, 2),(5, 4, 70, 30, 3),(5, 3, 50, 15, 4),(11, 6, 60, 20, 5)]
    for kernel, deviation, delta_tresh, min_area, i in parametrs:
        lr5(kernel, deviation, delta_tresh, min_area, i)