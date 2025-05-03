import os
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uuid
from pathlib import Path

from .detection import detect_objects
from .database import save_to_history, get_history, generate_report
from .models import DetectionResult

app = FastAPI(title="Sheep Counter System")

# Настройка статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Создание директорий для хранения загруженных и детектированных фото- видео-файлов
Path("static/uploads").mkdir(parents=True, exist_ok=True)
Path("static/results").mkdir(parents=True, exist_ok=True)

# Метод получения главной страницы сервиса
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Метод детектирования файла после нажатия на кнопку Process
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        file_ext = file.filename.split(".")[-1]
        filename = f"{uuid.uuid4()}.{file_ext}"
        file_path = f"static/uploads/{filename}"

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        is_video = file_ext.lower() in ['mp4', 'avi', 'mov', 'mkv']

        result = detect_objects(file_path, is_video=is_video)

        result_filename = f"result_{os.path.splitext(filename)[0]}.mp4" if is_video else f"result_{filename}"

        detection_result = DetectionResult(
            original_image=filename,
            processed_image=result_filename,
            sheep_count=result.count,
            processing_time=result.processing_time
        )
        save_to_history(detection_result)
        
        response_data = {
            "status": "success",
            "original": filename,
            "processed": result_filename,
            "count": result.count,
            "processing_time": result.processing_time,
            "is_video": is_video
        }
        
        if is_video:
            response_data.update({
                "min_count": result.min_count,
                "max_count": result.max_count,
                "total_frames": result.total_frames,
                "frame_stats": result.frame_stats
            })
        
        return JSONResponse(response_data)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# Метод перехода на страницу История запросов после нажатия на кнопку View full history
@app.get("/history")
async def history(request: Request):
    history_data = get_history()
    return templates.TemplateResponse(
        "history.html",
        {"request": request, "history": history_data}
    )

# Метод создания pdf-отчета после нажатия на кнопку 
@app.get("/report")
async def generate_report_endpoint():
    report_path = generate_report()
    if report_path:
        return FileResponse(
            report_path,
            media_type="application/octet-stream",
            filename=os.path.basename(report_path)
        )
    return JSONResponse({"status": "error", "message": "Report generation failed"})
    