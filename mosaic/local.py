from PIL import Image
from typing import Optional, List, Tuple


import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available


class LocalInferenceClient:

    def __init__(
        self, 
        model_name: str = "vidore/colqwen2-v1.0",
        device: str = "cuda:0"
    ):

        self.device = device
        self.model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,  # or "mps" if on Apple Silicon
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name)


    def encode_image(self, image: Image) -> List[List[float]]:
        processed_image = self.processor.process_images([image]).to(self.device)

        # Generate embedding
        with torch.inference_mode():
            embedding = self.model(**processed_image)
            return embedding.cpu().float().numpy().tolist()
        
    
    def encode_query(self, query: str) -> List[List[float]]:
        processed_query = self.processor.process_queries([query]).to(self.device)

        # Generate embedding
        with torch.inference_mode():
            embedding = self.model(**processed_query)
            return embedding.cpu().float().numpy().tolist()