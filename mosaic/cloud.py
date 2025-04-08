import requests

from PIL import Image
from typing import Optional, List, Tuple

from mosaic.schemas import ColPaliRequest
from mosaic.utils import base64_encode_image_list

from colpali_engine.models import ColQwen2Processor

class CloudInferenceClient:
    def __init__(
        self, 
        base_url: str,
        model_name: str = "vidore/colqwen2-v1.0",
    ):
    
        self.base_url = base_url

        # self.model_name = model_name
        # self.processor = ColQwen2Processor.from_pretrained(model_name)


    def encode_image(self, image: Image) -> List[List[float]]:
        
        # Generate embedding
        embedding = requests.post(
            f"{self.base_url}/image", 
            json={'inputs': base64_encode_image_list([image])}
        ).json()

        return embedding
        
    
    def encode_query(self, query: str) -> List[List[float]]:

        # Generate embedding
        embedding = requests.post(
            f"{self.base_url}/query", 
            json={'inputs': [query]}
        ).json()
        
        return embedding