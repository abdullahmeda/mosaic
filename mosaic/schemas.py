from pydantic import BaseModel
from typing import List, Optional, Dict


class ColPaliRequest(BaseModel):
    inputs: List[str]
    image_input: bool = False