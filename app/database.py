from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from .models import DetectionResult
import pandas as pd
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os

SQLALCHEMY_DATABASE_URL = "sqlite:///./sheep_counter.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class DetectionHistory(Base):
    __tablename__ = "detection_history"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    original_image = Column(String)
    processed_image = Column(String)
    sheep_count = Column(Integer)
    processing_time = Column(Float)

Base.metadata.create_all(bind=engine)

def save_to_history(result: DetectionResult):
    db = SessionLocal()
    try:
        db_item = DetectionHistory(
            original_image=result.original_image,
            processed_image=result.processed_image,
            sheep_count=result.sheep_count,
            processing_time=result.processing_time
        )
        db.add(db_item)
        db.commit()
    finally:
        db.close()

def get_history(limit: int = 100):
    db = SessionLocal()
    try:
        return db.query(DetectionHistory).order_by(DetectionHistory.timestamp.desc()).limit(limit).all()
    finally:
        db.close()

def generate_report():
    history = get_history(limit=1000)
    
    if not history:
        return None
    
    # Подготовка данных
    data = []
    for item in history:
        data.append([
            item.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            item.original_image,
            item.processed_image,
            item.sheep_count,
            f"{item.processing_time:.2f}s"
        ])
    
    # Генерация отчета
    report_dir = "static/reports"
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{report_dir}/sheep_report_{timestamp}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    
    elements = []
    elements.append(Paragraph("Sheep Detection Report", styles["Title"]))
    
    # Создание таблицы
    table_data = [["Timestamp", "Original", "Processed", "Count", "Time"]] + data
    table = Table(table_data)
    
    # Стиль таблицы
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), '#CCCCCC'),
        ('TEXTCOLOR', (0, 0), (-1, 0), '#000000'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), '#EEEEEE'),
        ('GRID', (0, 0), (-1, -1), 1, '#000000')
    ]))
    
    elements.append(table)
    doc.build(elements)
    
    return filename
