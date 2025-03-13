import uuid

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from qdrant_client.http import models
from qdrant_client import QdrantClient
from pdf2image import convert_from_path
from typing import Optional, List, Tuple

from mosaic.local import LocalInferenceClient
from mosaic.cloud import CloudInferenceClient

import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available

class Mosaic:

    def __init__(
        self, 
        collection_name: str,
        inference_client,
        binary_quantization: Optional[bool] = True
    ):
        
        self.inference_client = inference_client

        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(":memory:")

        if not self.collection_exists():
            result = self.create_collection(binary_quantization)
            assert result, f"Failed to create collection {self.collection_name}"


    @classmethod
    def from_pretrained(
        cls, 
        collection_name: str,
        device: str = "cuda:0",
        model_name: str = "vidore/colqwen2-v1.0",
        binary_quantization: Optional[bool] = True
    ):
        return cls(
            collection_name=collection_name,
            binary_quantization=binary_quantization,
            inference_client=LocalInferenceClient(
                model_name=model_name,
                device=device
            )
        )
    

    @classmethod
    def from_api(
        cls,
        collection_name: str,
        base_url: str,
        model_name: str = "vidore/colqwen2-v1.0",
        binary_quantization: Optional[bool] = True
    ):
        return cls(
            collection_name=collection_name,
            binary_quantization=binary_quantization,
            inference_client=CloudInferenceClient(
                base_url=base_url,
                model_name=model_name
            )
        )


    def collection_exists(self):
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        return self.collection_name in collection_names


    def create_collection(self, binary_quantization=True):
        return self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(
                        always_ram=True
                    ),
                ) if binary_quantization else None,
            )
        )


    def _add_to_index(
        self, 
        vectors,
        payloads
    ):
        
        assert len(vectors) == len(payloads), "Vectors and payloads must be of the same length"

        ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads
            )
        )
        

    def index_image(self, image: Image, metadata: dict):
        embedding = self.inference_client.encode_image(image)
        
        self._add_to_index(
            vectors=embedding,
            payloads=[metadata]
        )
        
    
    def index_file(self, path: Path, metadata: dict):
        images = convert_from_path(path)
        
        payloads = []
        embeddings = []
        for i, image in enumerate(tqdm(images), start=1):
            extended_metadata = {
                "path": str(path),
                "page": i,
                **metadata
            }
            payloads.append(extended_metadata)

            embedding = self.inference_client.encode_image(image)
            embedding = torch.tensor(embedding)

            embeddings.append(embedding)

        if embeddings:
            embeddings = torch.cat(embeddings, dim=0)
            
            self._add_to_index(
                vectors=embeddings,
                payloads=payloads
            )

        del images
        del embeddings

    
    def index_directory(self, path: Path):
        if path.is_dir():
            for file in path.iterdir():

                # Check if its a pdf
                if file.suffix == ".pdf":
                    self.index_file(file)

        else:
            raise ValueError("Path is not a directory")
        

    def search(self, query: str, top_k: int = 5):
        embedding = self.inference_client.encode_query(query)
        
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=embedding[0],
            limit=top_k
        )

        return results.points