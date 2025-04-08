from pydantic import BaseModel
from typing import List, Optional, Dict


class ColPaliRequest(BaseModel):
    inputs: List[str]
    image_input: bool = False


class Document(BaseModel):
    rank: int
    doc_id: str
    doc_abs_path: str
    page: int
    score: float
    image: Optional[str]
    metadata: Optional[Dict]