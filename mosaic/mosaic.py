import uuid

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from qdrant_client.http import models
from qdrant_client import QdrantClient
from pdf2image import convert_from_path
from typing import Optional, List, Tuple, Union, Dict, Any

import torch
from mosaic.utils import *
from mosaic.schemas import Document
from mosaic.local import LocalInferenceClient
from mosaic.cloud import CloudInferenceClient


class Mosaic:

    def __init__(
        self, 
        collection_name: str,
        inference_client,
        db_client: Optional[QdrantClient] = None,
        binary_quantization: Optional[bool] = True
    ):
        
        self.collection_name  = collection_name
        self.inference_client = inference_client

        self.qdrant_client = db_client or QdrantClient(":memory:")

        if not self.collection_exists():
            result = self._create_collection(binary_quantization)
            assert result, f"Failed to create collection {self.collection_name}"


    @classmethod
    def from_pretrained(
        cls, 
        collection_name: str,
        device: str = "cuda:0",
        db_client: Optional[QdrantClient] = None,
        model_name: str = "vidore/colqwen2-v1.0",
        binary_quantization: Optional[bool] = True
    ):
        return cls(
            collection_name=collection_name,
            db_client=db_client,
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
        db_client: Optional[QdrantClient] = None,
        model_name: str = "vidore/colqwen2-v1.0",
        binary_quantization: Optional[bool] = True
    ):
        return cls(
            collection_name=collection_name,
            db_client=db_client,
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


    def _create_collection(self, binary_quantization=True):
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
        vectors: List[List[List[float]]],
        payloads: List[Dict[str, Any]],
        batch_size: int = 16
    ):
        
        assert len(vectors) == len(payloads), "Vectors and payloads must be of the same length"

        for i in range(0, len(vectors), batch_size):
            batch_end = min(i + batch_size, len(vectors))

            # Slice the data for the current batch
            current_batch_vectors = vectors[i:batch_end]
            current_batch_payloads = payloads[i:batch_end]
            batch_len = len(current_batch_vectors)

            current_batch_ids = [str(uuid.uuid4()) for _ in range(batch_len)]

            try:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=models.Batch(
                        ids=current_batch_ids,
                        vectors=current_batch_vectors,
                        payloads=current_batch_payloads
                    ),
                    wait=True
                )

            except Exception as e:
                print(f"Failed to upsert points to collection '{self.collection_name}': {str(e)}")
        

    def index_image(
        self, 
        image: Image.Image, 
        metadata: Dict[str, Any] = None,
        store_img_bs64: Optional[bool] = True,
        max_image_dims: Tuple[int, int] = (1568, 1568)
    ):
        image_id = str(uuid.uuid4())

        max_img_height, max_img_width = max_image_dims
        image = resize_image(image, max_img_height, max_img_width)
        if store_img_bs64:
            bs64_image = base64_encode_image(image)

        embedding = self.inference_client.encode_image(image)

        payload = {
            "pdf_id": str(uuid.uuid4()), # Treat single image as a document
            "pdf_abs_path": None,        # No file path for direct image
            "page": 1,
            "image": bs64_image if store_img_bs64 else None,
            "metadata": metadata or {},
        }
        
        self._add_to_index(
            vectors=embedding,
            payloads=[payload]
        )

        return image_id
        
    
    def index_file(
        self, 
        path: Union[Path, str],
        metadata: Optional[dict] = {},
        store_img_bs64: Optional[bool] = True,
        max_image_dims: Tuple[int, int] = (1568, 1568)
    ):

        if type(path) == str:
            path = Path(path)
        abs_path = path.absolute()

        if not path.is_file():
            print(f"Path is not a file: {path}")
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.suffix.lower() != '.pdf':
            print(f"File is not a PDF: {path}")
            raise ValueError(f"File not a PDF: {path}")

        # --- Check for existing entries using count ---
        count_result = self.qdrant_client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="pdf_abs_path",
                        match=models.MatchValue(value=str(abs_path))
                    )
                ]
            ),
            exact=False # Set to True if you need the exact count, False is faster for just checking > 0
        )

        if count_result.count > 0:
            # TODO: Implement overwrite or skip logic
            print(f"File is already indexed: {path}")
            raise ValueError(f"File is already indexed: {path}")
        

        max_img_height, max_img_width = max_image_dims

        images = convert_from_path(path)
        images = resize_image_list(images, max_img_height, max_img_width)
        base64_images = [None] * len(images)

        if store_img_bs64:
            base64_images = base64_encode_image_list(images)

        pdf_id = str(uuid.uuid4())
        
        payloads = []
        embeddings = []
        for i, (image, bs64_img) in enumerate(tqdm(
            zip(images, base64_images), 
            total=len(images),
            desc=f"Indexing {str(path)}",
        ), start=1):
            
            extended_metadata = {
                "pdf_id": pdf_id,
                "pdf_abs_path": str(abs_path),
                "page": i,
                "image": bs64_img,
                "metadata": metadata,
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

        return pdf_id

    
    def index_directory(
        self, 
        path: Union[Path, str],
        metadata: Optional[dict] = {},
        store_img_bs64: Optional[bool] = True,
        max_image_dims: Tuple[int, int] = (1568, 1568),
    ):
        if type(path) == str:
            path = Path(path)

        if not path.is_dir():
            raise ValueError("Path is not a directory")
    
        docid2path = {}
        for file in path.iterdir():

            # Check if its a pdf
            if file.suffix == ".pdf":
                pdf_id = self.index_file(
                    path=file,
                    metadata=metadata,
                    store_img_bs64=store_img_bs64, 
                    max_image_dims=max_image_dims,
                )
                docid2path[pdf_id] = str(file.absolute())

            # Check if its an image file
            if file.suffix in [".png", ".jpg", ".jpeg"]:
                image = Image.open(file)
                image_id = self.index_image(
                    image=image,
                    metadata=metadata,
                    store_img_bs64=store_img_bs64,
                    max_image_dims=max_image_dims
                )
                docid2path[image_id] = str(file.absolute())
        
        return docid2path

    def search_text(
        self, 
        query: str, 
        top_k: int = 5
    ):
        embedding = self.inference_client.encode_query(query)
        
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=embedding[0],
            limit=top_k
        )

        documents = []
        for rank, point in enumerate(results.points, start=1):
            data = {
                'rank': rank, 
                'score': point.score, 
                **point.payload
            }
            documents.append(Document(**data))

        return documents
    

    def search_image(
        self, 
        image: Union[Image.Image, Path, str], 
        description: str = None, 
        top_k: int = 5
    ):
        if isinstance(image, (Path, str)):
            image = Image.open(image)

        embedding = self.inference_client.encode_image(image)
        if description:
            description_embedding = self.inference_client.encode_query(description)
            embedding = torch.cat([
                torch.tensor(embedding), torch.tensor(description_embedding)
            ], dim=1).tolist()
        
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=embedding[0],
            limit=top_k
        )

        documents = []
        for rank, point in enumerate(results.points, start=1):
            data = {
                'rank': rank, 
                'score': point.score, 
                **point.payload
            }
            documents.append(Document(**data))

        return documents