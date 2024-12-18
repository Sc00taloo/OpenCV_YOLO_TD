import cv2
import time
from mosse import Mosse, BoundingBox
from Summary_table import (recovery_rate_crst, lost_percentage_csrt, total_time_crst,
                           recovery_rate_median, lost_percentage_median, total_time_median,
                           recovery_rate_mosse, lost_percentage_mosse, total_time_mosse)

def trac(video_path, method):
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
        if method == 'MOSSE':
            tracker = cv2.legacy.TrackerMOSSE_create()
        elif method == 'CSRT':
            tracker = cv2.legacy.TrackerCSRT_create()
        elif method == 'My_MOSSE':
            tracker = Mosse()
        else:
            tracker = cv2.legacy.TrackerMedianFlow_create()

        if method == 'My_MOSSE':
            result_video_path = f"{method}_{video_path}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            result_video = cv2.VideoWriter(result_video_path, fourcc, 20.0, (w, h))
            start_time = time.time()
            lost_count = 0  # Счетчик потерянных объектов
            frame_count = 0  # Счетчик кадров
            recovery_count = 0  # Счетчик восстановлений
            successful_recovery = False  # Флаг для отслеживания успешного восстановления
            bbox = BoundingBox(bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2, bbox[2], bbox[3])
            tracker.init(frame, bbox)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                timer = cv2.getTickCount()
                # Обновляем трекер
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                bbox = tracker.update(frame)
                cv2.rectangle(frame,(bbox.left(), bbox.top()),(bbox.right(), bbox.bottom()),color=(0, 255, 0),thickness=2,)
                cv2.putText(frame, f"Method {method}", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                cv2.putText(frame, "FPS : " + str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

                result_video.write(frame)
                # Отображаем кадр
                cv2.imshow("Tracking", frame)
                # Выход при нажатии клавиши 'q'
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            end_time = time.time()
            total_time = end_time - start_time

            # lost_percentage = (lost_count / frame_count) * 100 if frame_count > 0 else 0
            # print(f"Частота потери изображения: {lost_percentage:.2f}%")
            # lost_percentage_median.append(f"{lost_percentage:.2f}")
            # print(f"Время работы метода: {total_time:.3f} секунд")
            # total_time_median.append(f"{total_time:.3f}")
            # recovery_rate = (recovery_count / lost_count) * 100 if lost_count > 0 else 0
            # recovery_rate_median.append(f"{recovery_rate:.2f}")
            # print(f"Частота восстановления: {recovery_rate:.2f}%")

            cap.release()
            result_video.release()
            cv2.destroyAllWindows()
        else:
            # Инициализируем трекер с первым кадром и выбранной областью
            tracker.init(frame , bbox)
            result_video_path = f"{method}_{video_path}"
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
                    cv2.putText(frame, "Fail", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.putText(frame, f"Method {method}", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                cv2.putText(frame, "FPS : " + str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

                result_video.write(frame)
                # Отображаем кадр
                cv2.imshow("Tracking", frame)
                # Выход при нажатии клавиши 'q'
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            end_time = time.time()
            total_time = end_time - start_time

            # Вычисляем параметры
            # if method == 'MOSSE':
            #     lost_percentage = (lost_count / frame_count) * 100 if frame_count > 0 else 0
            #     print(f"Частота потери изображения: {lost_percentage:.2f}%")
            #     lost_percentage_mosse.append(f"{lost_percentage:.2f}")
            #     print(f"Время работы метода: {total_time:.3f} секунд")
            #     total_time_mosse.append(f"{total_time:.3f}")
            #     recovery_rate = (recovery_count / lost_count) * 100 if lost_count > 0 else 0
            #     recovery_rate_mosse.append(f"{recovery_rate:.2f}")
            #     print(f"Частота восстановления: {recovery_rate:.2f}%")
            # elif method == 'CSRT':
            #     lost_percentage = (lost_count / frame_count) * 100 if frame_count > 0 else 0
            #     print(f"Частота потери изображения: {lost_percentage:.2f}%")
            #     lost_percentage_csrt.append(f"{lost_percentage:.2f}")
            #     print(f"Время работы метода: {total_time:.3f} секунд")
            #     total_time_crst.append(f"{total_time:.3f}")
            #     recovery_rate = (recovery_count / lost_count) * 100 if lost_count > 0 else 0
            #     recovery_rate_crst.append(f"{recovery_rate:.2f}")
            #     print(f"Частота восстановления: {recovery_rate:.2f}%")
            # else:
            #     lost_percentage = (lost_count / frame_count) * 100 if frame_count > 0 else 0
            #     print(f"Частота потери изображения: {lost_percentage:.2f}%")
            #     lost_percentage_median.append(f"{lost_percentage:.2f}")
            #     print(f"Время работы метода: {total_time:.3f} секунд")
            #     total_time_median.append(f"{total_time:.3f}")
            #     recovery_rate = (recovery_count / lost_count) * 100 if lost_count > 0 else 0
            #     recovery_rate_median.append(f"{recovery_rate:.2f}")
            #     print(f"Частота восстановления: {recovery_rate:.2f}%")

            # Освобождаем ресурсы
            cap.release()
            result_video.release()
            cv2.destroyAllWindows()

def tracking(method):
    video_paths = ['Car1.mp4', 'Car2.mp4','Car3.mp4','Car4.mp4', 'Car5.mp4']
    for video in video_paths:
        trac(video, method)