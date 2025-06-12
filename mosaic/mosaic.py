import uuid

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

    def index_image(
        self,
        image: Image.Image,
        metadata: Dict[str, Any] = None,
        store_img_bs64: Optional[bool] = True,
        max_image_dims: Tuple[int, int] = (1568, 1568),
    ):
        image_id = str(uuid.uuid4())

        max_img_height, max_img_width = max_image_dims
        image = resize_image(image, max_img_height, max_img_width)
        if store_img_bs64:
            bs64_image = base64_encode_image(image)

        embedding = self.inference_client.encode_image(image)

        payload = {
            "pdf_id": str(uuid.uuid4()),  # Treat single image as a document
            "pdf_abs_path": None,  # No file path for direct image
            "page_number": 1,
            "base64_image": bs64_image if store_img_bs64 else None,
            "metadata": metadata or {},
        }

        self._add_to_index(vectors=embedding, payloads=[payload])

        return image_id

    def index_file(
        self,
        path: Union[Path, str],
        metadata: Optional[dict] = {},
        store_img_bs64: Optional[bool] = True,
        max_image_dims: Tuple[int, int] = (1568, 1568),
        avoid_file_existence_check: Optional[bool] = False,
        pdf_output_path: Optional[Union[Path, str]] = None,
    ):
        """Index a file by converting it to PDF if necessary and processing it.
        
        Args:
            path: Path to the file to index
            metadata: Additional metadata to store with the document
            store_img_bs64: Whether to store base64-encoded images
            max_image_dims: Maximum image dimensions (height, width)
            avoid_file_existence_check: Skip checking if file is already indexed
            pdf_output_path: Path where converted PDF should be saved (for non-PDF files).
                           If None, saves alongside the original file with .pdf extension.
        
        Returns:
            Document ID of the indexed file
        """
        if type(path) is str:
            path = Path(path)
        abs_path = path.absolute()

        if not path.is_file():
            print(f"Path is not a file: {path}")
            raise FileNotFoundError(f"File not found: {path}")

        # Check if file is supported (PDF or convertible formats)
        file_suffix = path.suffix.lower()
        if file_suffix not in {".pdf"} | ALLOWED_EXT:
            print(f"File type not supported: {path}")
            raise ValueError(f"File type {file_suffix} not supported. Supported types: .pdf, {', '.join(sorted(ALLOWED_EXT))}")

        # Convert to PDF if necessary
        pdf_path = path
        
        if file_suffix in ALLOWED_EXT:
            print(f"Converting {path} to PDF using Gotenberg...")
            if pdf_output_path:
                if type(pdf_output_path) is str:
                    pdf_output_path = Path(pdf_output_path)
                pdf_path = self._convert_to_pdf(path, pdf_output_path)
            else:
                pdf_path = self._convert_to_pdf(path)
            print(f"Converted PDF saved to: {pdf_path}")

        if not avoid_file_existence_check:
            print(f"Checking for existing entries: {path}")
            # --- Check for existing entries using count ---
            count_result = self.qdrant_client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="pdf_abs_path", match=models.MatchValue(value=str(abs_path))
                        )
                    ]
                ),
                exact=True,  # Set to True for exact matching to avoid false positives
            )

            if count_result.count > 0:
                # TODO: Implement overwrite or skip logic
                print(f"File is already indexed: {path}")
                raise ValueError(f"File is already indexed: {path}")

        max_img_height, max_img_width = max_image_dims

        images = convert_from_path(pdf_path)
        images = resize_image_list(images, max_img_height, max_img_width)
        base64_images = [None] * len(images)

        if store_img_bs64:
            base64_images = base64_encode_image_list(images)

        pdf_id = str(uuid.uuid4())

        payloads = []
        embeddings = []
        for i, (image, bs64_img) in enumerate(
            tqdm(
                zip(images, base64_images),
                total=len(images),
                desc=f"Indexing {str(path)}",
            ),
            start=1,
        ):
            extended_metadata = {
                "pdf_id": pdf_id,
                "pdf_abs_path": str(abs_path),  # Store original file path
                "converted_pdf_path": str(pdf_path) if pdf_path != path else None,  # Store converted PDF path if different
                "page_number": i,
                "base64_image": bs64_img,
                "metadata": metadata,
            }
            payloads.append(extended_metadata)

            embedding = self.inference_client.encode_image(image)
            embedding = np.array(embedding)

            embeddings.append(embedding)

        if embeddings:
            embeddings = np.concatenate(embeddings, axis=0)

            self._add_to_index(vectors=embeddings, payloads=payloads)

        del images
        del embeddings

        return pdf_id

    def index_directory(
        self,
        path: Union[Path, str],
        metadata: Optional[dict] = {},
        store_img_bs64: Optional[bool] = True,
        max_image_dims: Tuple[int, int] = (1568, 1568),
        pdf_output_dir: Optional[Union[Path, str]] = None,
    ):
        """Index all supported files in a directory.
        
        Args:
            path: Path to the directory to index
            metadata: Additional metadata to store with documents
            store_img_bs64: Whether to store base64-encoded images
            max_image_dims: Maximum image dimensions (height, width)
            pdf_output_dir: Directory where converted PDFs should be saved.
                          If None, saves alongside the original files.
        
        Returns:
            Dictionary mapping document IDs to file paths
        """
        if type(path) is str:
            path = Path(path)

        if not path.is_dir():
            raise ValueError("Path is not a directory")

        # Prepare PDF output directory if specified
        if pdf_output_dir:
            if type(pdf_output_dir) is str:
                pdf_output_dir = Path(pdf_output_dir)
            pdf_output_dir.mkdir(parents=True, exist_ok=True)

        docid2path = {}
        for file in path.iterdir():
            # Check if its a pdf or supported document type
            if file.suffix.lower() in {".pdf"} | ALLOWED_EXT:
                try:
                    # Generate PDF output path if directory is specified
                    pdf_output_path = None
                    if pdf_output_dir and file.suffix.lower() in ALLOWED_EXT:
                        pdf_output_path = pdf_output_dir / f"{file.stem}.pdf"
                    
                    pdf_id = self.index_file(
                        path=file,
                        metadata=metadata,
                        store_img_bs64=store_img_bs64,
                        max_image_dims=max_image_dims,
                        pdf_output_path=pdf_output_path,
                    )
                    docid2path[pdf_id] = str(file.absolute())
                except Exception as e:
                    print(f"Failed to index {file}: {str(e)}")
                    continue

            # Check if its an image file
            elif file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                try:
                    image = Image.open(file)
                    image_id = self.index_image(
                        image=image,
                        metadata=metadata,
                        store_img_bs64=store_img_bs64,
                        max_image_dims=max_image_dims,
                    )
                    docid2path[image_id] = str(file.absolute())
                except Exception as e:
                    print(f"Failed to index {file}: {str(e)}")
                    continue
            else:
                print(f"Skipping {file.name} (unsupported file type)")

        return docid2path

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

    def search_image(
        self,
        image: Union[Image.Image, Path, str],
        description: str = None,
        top_k: int = 5,
    ):
        if isinstance(image, (Path, str)):
            image = Image.open(image)

        embedding = self.inference_client.encode_image(image)
        if description:
            description_embedding = self.inference_client.encode_query(description)
            embedding = np.concatenate(
                [np.array(embedding), np.array(description_embedding)], axis=1
            ).tolist()

        results = self.qdrant_client.query_points(
            collection_name=self.collection_name, query=embedding[0], limit=top_k
        )

        documents = []
        for rank, point in enumerate(results.points, start=1):
            data = {"rank": rank, "score": point.score, **point.payload}
            documents.append(Document(**data))

        return documents
