from ultralytics import YOLO
import cv2
from PIL import Image
import time
from typing import NamedTuple, List, Tuple
import os
import numpy as np

class DetectionResult(NamedTuple):
    image: Image.Image
    count: int  # количество овец на изображении или среднее количество овец в кадре
    processing_time: float  # время обработки медиа файла
    frame_stats: List[Tuple[int, int, float]] = None  # для построения графика зависимости количества объектов от кадров
    min_count: int = 0  # минимальное количество овец в кадре
    max_count: int = 0  # максимальное количество овец в кадре
    total_frames: int = 0   # общее количество кадров в видео

# Загрузка модели
model = YOLO('yolov8n.pt')

def detect_objects(image_path: str, is_video: bool = False) -> DetectionResult:
    start_time = time.time()
    
    if not is_video:
        # Обработка изображения
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Детекция объектов
        results = model(img_rgb)
        
        # Подсчет овец
        sheep_classes = ['sheep']
        sheep_count = 0
        for result in results:
            for box in result.boxes:
                class_name = model.names[int(box.cls)]
                if class_name in sheep_classes:
                    sheep_count += 1
        
        plotted_img = results[0].plot()
        plotted_img_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(plotted_img_rgb)

        # Сохранение обработанного изображения
        result_filename = f"result_{os.path.basename(image_path)}"
        result_path = f"static/results/{result_filename}"
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        pil_img.save(result_path, quality=95)
        
        processing_time = time.time() - start_time
        
        return DetectionResult(
            image=pil_img,
            count=sheep_count,
            processing_time=round(processing_time, 2)
        )
    
    else:
        # Обработка видео
        cap = cv2.VideoCapture(image_path)
        if not cap.isOpened():
            raise ValueError("Could not open video source")
        
        frame_stats = []
        processed_frames = []
        frame_count = 0
        counts = []

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = f"static/results/result_{os.path.basename(image_path)}"

        fourcc = None
        for codec in ['avc1', 'h264', 'mp4v']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            if fourcc != -1:
                break
        
        if fourcc == -1:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        if not output_path.lower().endswith('.mp4'):
            output_path = os.path.splitext(output_path)[0] + '.mp4'
            
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError("Could not create video writer")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            frame_start = time.time()
            
            # Детекция объектов
            results = model(frame)
            
            # Подсчет овец
            sheep_count = 0
            for result in results:
                for box in result.boxes:
                    class_name = model.names[int(box.cls)]
                    if class_name == 'sheep':
                        sheep_count += 1
            
            counts.append(sheep_count)
            
            # Визуализация
            plotted_frame = results[0].plot()
            
            # Конвертация в RGB перед записью
            plotted_frame_rgb = cv2.cvtColor(plotted_frame, cv2.COLOR_BGR2RGB)
            out.write(plotted_frame_rgb)
            
            frame_stats.append((frame_count, sheep_count, time.time() - frame_start))

            if frame_count % 10 == 0:
                processed_frames.append(plotted_frame_rgb)
        
        cap.release()
        out.release()
        
        # Рассчет статистики
        if counts:
            avg_count = round(np.mean(counts))
            min_count = min(counts)
            max_count = max(counts)
        else:
            avg_count = min_count = max_count = 0
        
        processing_time = time.time() - start_time

        last_frame = None
        if processed_frames:
            last_frame = Image.fromarray(processed_frames[-1])
        
        return DetectionResult(
            image=last_frame,
            count=avg_count,
            processing_time=round(processing_time, 2),
            frame_stats=frame_stats,
            min_count=min_count,
            max_count=max_count,
            total_frames=frame_count
        )
    