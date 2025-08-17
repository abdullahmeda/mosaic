import uuid
import concurrent.futures
import threading

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from qdrant_client.http import models
from qdrant_client import QdrantClient
from pdf2image import convert_from_path
from typing import Optional, List, Tuple, Union, Dict, Any
from gotenberg_client import GotenbergClient

import numpy as np
from mosaic.schemas import Document
from mosaic.utils import (
    base64_encode_image_list,
    base64_encode_image,
    resize_image,
    resize_image_list,
)

# Supported file extensions for Gotenberg conversion
ALLOWED_EXT = {
    ".txt", ".rtf", ".doc", ".docx", ".odt",
    ".ppt", ".pptx", ".odp"
}


class Mosaic:
    def __init__(
        self,
        collection_name: str,
        inference_client: Any,
        db_client: Optional[QdrantClient] = None,
        binary_quantization: Optional[bool] = True,
        gotenberg_url: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.inference_client = inference_client
        self.gotenberg_url = gotenberg_url
        self._index_lock = threading.Lock()

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
        binary_quantization: Optional[bool] = True,
        gotenberg_url: Optional[str] = None,
    ):
        from mosaic.local import LocalInferenceClient

        return cls(
            collection_name=collection_name,
            db_client=db_client,
            binary_quantization=binary_quantization,
            gotenberg_url=gotenberg_url,
            inference_client=LocalInferenceClient(model_name=model_name, device=device),
        )

    @classmethod
    def from_api(
        cls,
        collection_name: str,
        base_url: str,
        db_client: Optional[QdrantClient] = None,
        model_name: str = "vidore/colqwen2-v1.0",
        binary_quantization: Optional[bool] = True,
        gotenberg_url: Optional[str] = None,
    ):
        from mosaic.cloud import CloudInferenceClient
        
        return cls(
            collection_name=collection_name,
            db_client=db_client,
            binary_quantization=binary_quantization,
            gotenberg_url=gotenberg_url,
            inference_client=CloudInferenceClient(
                base_url=base_url, model_name=model_name
            ),
        )

    def collection_exists(self):
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        return self.collection_name in collection_names

    def _convert_to_pdf(self, file_path: Path, output_pdf_path: Optional[Path] = None) -> Path:
        """Convert a file to PDF using Gotenberg.
        
        Args:
            file_path: Path to the file to convert
            output_pdf_path: Path where the converted PDF should be saved. 
                           If None, saves alongside the original file with .pdf extension.
        
        Returns:
            Path to the converted PDF file
        """
        if not self.gotenberg_url:
            raise ValueError(
                f"Gotenberg URL not provided. Cannot convert {file_path.suffix} files. "
                "Please provide gotenberg_url when initializing Mosaic."
            )
        
        if file_path.suffix.lower() not in ALLOWED_EXT:
            raise ValueError(f"File extension {file_path.suffix} not supported for conversion")
        
        # Auto-generate output path if not provided
        if output_pdf_path is None:
            output_pdf_path = file_path.with_suffix(".pdf")
        
        # Ensure output directory exists
        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with GotenbergClient(self.gotenberg_url) as client:
                with client.libre_office.to_pdf() as route:
                    route.convert(file_path)
                    response = route.run()
                    response.to_file(output_pdf_path)
            
            return output_pdf_path
        except Exception as e:
            raise RuntimeError(f"Failed to convert {file_path} to PDF: {str(e)}")

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
                    binary=models.BinaryQuantizationConfig(always_ram=True),
                )
                if binary_quantization
                else None,
            ),
        )

    def _add_to_index(
        self,
        vectors: List[List[List[float]]],
        payloads: List[Dict[str, Any]],
        batch_size: int = 16,
    ):
        assert len(vectors) == len(payloads), (
            "Vectors and payloads must be of the same length"
        )

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
                        payloads=current_batch_payloads,
                    ),
                    wait=True,
                )

            except Exception as e:
                print(
                    f"Failed to upsert points to collection '{self.collection_name}': {str(e)}"
                )

    def index_file(
        self,
        file_id: str,
        file_path: Path,
        metadata: Optional[dict] = {},
        store_img_bs64: Optional[bool] = True,
        max_image_dims: Tuple[int, int] = (1568, 1568),
        avoid_file_existence_check: Optional[bool] = False,
        pdf_output_path: Optional[Union[Path, str]] = None,
    ):
        file_path = file_path.absolute()

        if not file_path.is_file():
            print(f"Path is not a file: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        max_img_height, max_img_width = max_image_dims

        images = convert_from_path(file_path)
        images = resize_image_list(images, max_img_height, max_img_width)
        base64_images = [None] * len(images)

        if store_img_bs64:
            base64_images = base64_encode_image_list(images)

        def process_page(args):
            page_number, (image, bs64_img) = args
            extended_metadata = {
                "file_id": file_id,
                "file_abs_path": str(file_path),  # Store original file path
                "page_number": page_number,
                "base64_image": bs64_img,
                "metadata": metadata,
            }
            embedding = self.inference_client.encode_image(image)
            embedding = np.array(embedding)
            return extended_metadata, embedding

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            args_list = enumerate(zip(images, base64_images), start=1)
            results = list(
                tqdm(
                    executor.map(process_page, args_list),
                    total=len(images),
                    desc=f"Indexing {str(file_path)}",
                )
            )

        if results:
            payloads, embeddings = zip(*results)
            embeddings = list(embeddings)
            payloads = list(payloads)
        else:
            payloads, embeddings = [], []

        if embeddings:
            embeddings = np.concatenate(embeddings, axis=0)

            with self._index_lock:
                self._add_to_index(vectors=embeddings, payloads=payloads)

        del images
        del embeddings


    def search_text(self, query: str, top_k: int = 5):
        embedding = self.inference_client.encode_query(query)

        results = self.qdrant_client.query_points(
            collection_name=self.collection_name, query=embedding[0], limit=top_k
        )

        documents = []
        for rank, point in enumerate(results.points, start=1):
            data = {"rank": rank, "score": point.score, **point.payload}
            documents.append(Document(**data))

        return documents

    def remove_file(self, relative_file_path: str):
        """Remove all documents from the index that match the given relative file path.
        
        Args:
            relative_file_path: The relative file path stored in metadata['metadata']['relative_file_path']
        
        Returns:
            pdf_id of the removed documents, or None if no documents found
        """
        
        # Count total documents for confirmation
        count_result = self.qdrant_client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.relative_file_path", 
                        match=models.MatchValue(value=relative_file_path)
                    )
                ]
            ),
            exact=True,
        )
        
        # Delete the documents
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.relative_file_path",
                            match=models.MatchValue(value=relative_file_path)
                        )
                    ]
                )
            ),
        )
        
        print(f"Removed {count_result.count} documents with relative_file_path: {relative_file_path}")
        return count_result.count
