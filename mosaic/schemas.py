from pydantic import BaseModel
from typing import Optional, Dict


class Document(BaseModel):
    pdf_id: str
    page_number: int
    rank: Optional[int]
    score: Optional[float]
    metadata: Optional[Dict]
    base64_image: Optional[str]
    pdf_abs_path: Optional[str]
    converted_pdf_path: Optional[str] = None
