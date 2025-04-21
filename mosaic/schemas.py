from pydantic import BaseModel
from typing import List, Optional, Dict


class Document(BaseModel):
    pdf_id: str
    page: int
    rank: Optional[int]
    image: Optional[str]
    score: Optional[float]
    metadata: Optional[Dict]
    pdf_abs_path: Optional[str]