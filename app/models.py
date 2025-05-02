from pydantic import BaseModel
from typing import NamedTuple
from datetime import datetime

class DetectionResult(BaseModel):
    original_image: str
    processed_image: str
    sheep_count: int
    processing_time: float

class DetectionInput(BaseModel):
    image_path: str
    camera_mode: bool = False

class DetectionResponse(BaseModel):
    status: str
    original: str
    processed: str
    count: int
    processing_time: float
    