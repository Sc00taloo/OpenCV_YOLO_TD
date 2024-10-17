import cv2
import time
from Summary_table import recovery_rate_crst, lost_percentage_csrt, total_time_crst

def csrt_trac(video_path):
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Читаем первый кадр
    ret, frame = cap.read()
    if not ret:
        print("Не удалось прочитать видео")
        return
    # Выбираем область для трекинга
    bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
    if bbox[2] > 0 and bbox[3] > 0:
        cv2.destroyWindow("Frame")
        # Создаем трекер MOSSE
        tracker = cv2.legacy.TrackerCSRT_create()
        # Инициализируем трекер с первым кадром и выбранной областью
        tracker.init(frame, bbox)
        result_video_path = f"CSRT_{video_path}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        result_video = cv2.VideoWriter(result_video_path, fourcc, 20.0, (w, h))
        start_time = time.time()
        lost_count = 0  # Счетчик потерянных объектов
        frame_count = 0  # Счетчик кадров
        recovery_count = 0  # Счетчик восстановлений
        successful_recovery = False  # Флаг для отслеживания успешного восстановления

        while True:
            # Читаем следующий кадр
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            timer = cv2.getTickCount()
            # Обновляем трекер
            success, bbox = tracker.update(frame)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            # Проверяем, вышел ли объект за границы
            if not success or bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > w or bbox[1] + bbox[3] > h:
                lost_count += 1
                if successful_recovery:  # Если объект уже был восстановлен, увеличиваем счетчик
                    recovery_count += 1
                successful_recovery = False  # Сбрасываем флаг
            else:
                successful_recovery = True  # Объект в пределах видимости

            if success:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                cv2.putText(frame, "Fail", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            cv2.putText(frame, "CSRT Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            result_video.write(frame)
            # Отображаем кадр
            cv2.imshow("Tracking", frame)
            # Выход при нажатии клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        end_time = time.time()
        total_time = end_time - start_time

        # Вычисляем параметры
        lost_percentage = (lost_count / frame_count) * 100 if frame_count > 0 else 0
        print(f"Частота потери изображения: {lost_percentage:.2f}%")
        lost_percentage_csrt.append(f"{lost_percentage:.2f}")
        print(f"Время работы метода: {total_time:.3f} секунд")
        total_time_crst.append(f"{total_time:.3f}")
        recovery_rate = (recovery_count / lost_count) * 100 if lost_count > 0 else 0
        recovery_rate_crst.append(f"{recovery_rate:.2f}")
        print(f"Частота восстановления: {recovery_rate:.2f}%")

        # Освобождаем ресурсы
        cap.release()
        result_video.release()
        cv2.destroyAllWindows()

def csrt_tracking():
    video_paths = ['Car1.mp4', 'Car2.mp4', 'Car3.mp4', 'Car4.mp4', 'Car5.mp4']
    for video in video_paths:
        csrt_trac(video)