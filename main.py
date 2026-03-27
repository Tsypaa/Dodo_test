from ultralytics import YOLO
import pandas as pd
import cv2
import argparse
import numpy as np
from collections import deque
import time

def parse_args():
    """Аргументы"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Веса модели")
    parser.add_argument("--video", type=str, default="video.mp4", help="Путь к видео")
    parser.add_argument("--output", type=str, default="output.mp4", help="Путь к выходному видео")
    parser.add_argument("--confidence", type=float, default=0.5, help="Порог вероятности для модели")

    return parser.parse_args()

def select_table(frame, window_name="Select table", max_width=1200, max_height=800):
    """
    Выбор стола вручную.
    frame: кадр видео
    window_name: имя окна
    max_width: максимальная ширина (для удобства выбора на больших видео)
    max_height: максимальная высота (для удобства выбора на больших видео)
    Управление:
    - Нажмите ESC для отмены
    - Нажмите ENTER для завершения
    - Нажмите BACKSPACE для удаления последней точки
    - Нажмите R для сброса всё
    - ЛКМ: добавить точку
    """

    points = []
    height, width = frame.shape[:2]
    
    # Масштабируем, если изображение больше максимальных размеров
    scale = 1.0
    if width > max_width or height > max_height:
        scale_x = max_width / width
        scale_y = max_height / height
        scale = min(scale_x, scale_y)
        new_width = int(width * scale)
        new_height = int(height * scale)
        display_frame = cv2.resize(frame, (new_width, new_height))
    else:
        display_frame = frame.copy()
        scale = 1.0

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"Точка {len(points)}: ({x}, {y})")

    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp = display_frame.copy()
        
        # Рисуем точки и линии
        for i, (x, y) in enumerate(points):
            cv2.circle(temp, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(temp, str(i+1), (x+8, y-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        if len(points) >= 2:
            for i in range(len(points)-1):
                cv2.line(temp, points[i], points[i+1], (0, 255, 0), 2)
        
        if len(points) == 4:
            cv2.line(temp, points[3], points[0], (0, 255, 0), 2)
            cv2.putText(temp, "ENTER", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, temp)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC - отмена
            cv2.destroyAllWindows()
            return None
        elif key == 13 and len(points) == 4:  # ENTER - завершить
            break
        elif key == 8 and points:  # BACKSPACE - удалить последнюю
            points.pop()
        elif key == ord('r') and points:  # R - сбросить всё
            points = []

    if len(points) != 4:
        print("Ошибка: нужно выбрать 4 точки!")
        return None
    
    cv2.destroyAllWindows()

    # Возвращаем координаты в исходном масштабе
    if scale != 1.0:
        points_array = np.array(points, dtype=np.float32)
        points = points_array / scale
        
    return points.astype(np.int32)

def person_in_area(detections, points):
    """
    Проверка, что человек находится внутри стола
    detections: список bbox [x1, y1, x2, y2, conf, class_id]
    roi: координаты прямоугольника
    frame_width: ширина кадра
    frame_height: высота кадра
    Возвращает True, если человек находится внутри стола, иначе False
    """
    for *box, conf, cls in detections:
        x, y, x2, y2 = box
        x_person = (x + x2) // 2
        y_person = (y + y2) // 2
        if cv2.pointPolygonTest(points, (x_person, y_person), False) >= 0:
            return True
    return False

def main():
    args = parse_args() # Получаем аргументы

    model = YOLO("yolov8n.pt") # Загружаем модель

    cap = cv2.VideoCapture(args.video) # Открываем видео

    # Проверка на доступ к видео
    if not cap.isOpened():
        print("Не удалось открыть видео")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Ширина кадра
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Высота кадра
    fps = cap.get(cv2.CAP_PROP_FPS) # Частота кадра
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)) # Создаем видеофайл

    # Таймер для подтверждения ухода
    disappearance_time = 0.0

    ret, frame = cap.read() # Читаем первый кадр

    points = select_table(frame) # Выбираем стол по первому кадру

    print(f"Выбраннный стол: {points}")

    # Состояние
    STATE_FREE = "Свободен"
    STATE_OCCUPIED = "Занят"
    state = STATE_FREE

    # Сколько секунд ждать, чтобы убедиться, что человек ушел (Время ожидания компенсировано)
    wait_time = 7

    # Список времени между событиями
    processing_times = []

    # Флаг для проверки подхода (появления человека) - для предотвращения множественных событий при заполнении буфера
    approach = False

    # Создаем буфер c максимальным размером 16
    occupied_buffer = deque(maxlen=16)
    
    # Для детекции начала заполнения буфера
    previous_occupancy = 0

    events = [] # Хранение событий: список кортежей (время, событие)

    frame_idx = 0 # Индекс кадра

    while ret:
        start_time = time.time()  # начало обработки кадра

        ret, frame = cap.read()
        if not ret:
            cv2.destroyAllWindows() # Закрываем окна, если не удалось прочитать кадр
            break

        timestamp = frame_idx / fps  # секунды от начала видео

        # Детекция людей
        results = model(frame, conf=args.confidence, classes=[0], verbose=False)

        # Получаем bbox
        detections = results[0].boxes.data.cpu().numpy()  # [x1,y1,x2,y2,conf,cls]

        # Проверка на наличие человека
        person_present = person_in_area(detections, points)

        # Обновляем буфер
        occupied_buffer.append(person_present)

                
        # Текущая заполненность буфера
        current_occupancy = sum(occupied_buffer)

        # Условия
        # Стабильное состояние (большинство кадров занято)
        stable_occupied = current_occupancy >= len(occupied_buffer) // 2

        # Подход (в буфере 5 кадров занято)
        is_approach = (
            previous_occupancy <= 5 and # Условие 1: было пусто
            current_occupancy > previous_occupancy and # Условие 2: началось заполнение
            current_occupancy < len(occupied_buffer) - 5 # Условие 3: ещё не стабилизировалось
        )

        # Если первые два кадра заняты и состояние = пусто, то добавляем событие подход
        if is_approach and state == STATE_FREE and approach == False:
            events.append((timestamp, "Подход"))
            print(f"[{timestamp:.2f}s] ПОДОШЕЛ ЧЕЛОВЕК")
            approach = True # Флаг подхода установлен, чтобы не генерировать множественные события при заполнении буфера

        # Если большинство кадров занято и состояние = пусто, то меняем состояние на занято
        if stable_occupied and state == STATE_FREE:
            state = STATE_OCCUPIED
            approach = False # Сброс флага подхода, так как мы уже в состоянии занято
            events.append((timestamp, "Занято"))
            print(f"[{timestamp:.2f}s] СТОЛ ЗАНЯТ")
        
        # Обновляем время отсчета
        if stable_occupied:
            disappearance_time = 0.0

        # Проверка на то, ушел ли человек
        elif not (stable_occupied) and state == STATE_OCCUPIED:
            if disappearance_time == 0.0:
                disappearance_time = timestamp
                print(f"[{timestamp:.2f}s] ЧЕЛОВЕК ПРОПАЛ, жду {wait_time:.2f}с...")
            elif timestamp - disappearance_time >= wait_time:
                state = STATE_FREE
                events.append((timestamp - wait_time, "Свободно")) # Компенсируем время проверки
                print(f"[{timestamp:.2f}s] СТОЛ СВОБОДЕН")
                disappearance_time = 0.0

        # Сохраняем предыдущие значения
        previous_occupancy = current_occupancy

        # Визуализация
        # Рисуем область с цветом в зависимости от состояния
        # Зеленый - пусто, красный - занято
        if state == STATE_FREE:
            color = (0, 255, 0)
            cv2.putText(frame, f"State: Free", (points[0][0], points[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            color = (0, 0, 255)
            cv2.putText(frame, f"State: Occupied", (points[0][0], points[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.polylines(frame, [points.reshape((-1, 1, 2))], True, color, 3)
        
        # Дополнительно рисуем bounding boxes всех людей
        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        
        # Отображаем кадр
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC для выхода
            cv2.destroyWindow("Frame")
        
        # Записываем кадр
        out.write(frame)

        # Записываем время обработки кадра
        processing_time = time.time() - start_time
        processing_times.append(processing_time)

        # Обновляем индекс кадра
        frame_idx += 1

    # Закрываем видео
    cap.release()

    # Записываем видео
    out.release()

    # Закрываем все окна
    cv2.destroyAllWindows()

    if len(events) == 0:
        print("Нет зафиксированных событий")
        return
    
    # Создаем таблицу ивентов и сохраняем
    df_events = pd.DataFrame(events, columns=["Time", "Event"])
    df_events.to_csv("events.csv", index=False)

    # Берем времена событий Свободно и Занято
    free_times = df_events[df_events["Event"] == "Свободно"]["Time"].values
    approach_times = df_events[df_events["Event"] == "Занято"]["Time"].values
    
    delays = []
    # Расчет среднего времени между уходом и следующим подходом
    for approach_time in approach_times:
        prev_free = free_times[free_times < approach_time]
        if len(prev_free) > 0:
            last_empty = prev_free[-1]
            delay = approach_time - last_empty
            delays.append(delay)
    # Выводим статистику и сохраняем в файл
    if delays:
        mean_delay = np.mean(delays)
        mean_ms = np.mean(processing_times) * 1000
        print("Статистика")
        print(f"Среднее время между уходом гостя и подходом следующего человека, который занял стол: {mean_delay:.2f} секунд")
        print(f"Количество подходов: {len(delays)}")
        print(f"Задержка видео: {(1/fps - mean_ms/1000)*1000:.2f} мс/кадр")

        with open("report.txt", "w", encoding='utf-8') as f:
            f.write(f"Среднее время между уходом гостя и подходом следующего человека, который занял стол: {mean_delay:.2f} секунд\n")
            f.write(f"Количество подходов: {len(delays)}\n")
            f.write(f"Задержка видео: {(1/fps - mean_ms/1000)*1000:.2f} мс/кадр\n")
    else:
        print("Нет подходов")
            
if __name__ == "__main__":
    main()