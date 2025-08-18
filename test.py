from pathlib import Path
from mosaic import Mosaic
from pdf2image import convert_from_path

mosaic = Mosaic.from_api('test_big_1111', "http://172.25.44.10:8000/colqwen2/v1/embeddings", binary_quantization=False)

pdf_path = Path('/home/abdullah/Projects/mosaic/docs/Continued debt reduction _ Annual report - Genève Aéroport.pdf')

images = convert_from_path(
    pdf_path,
    dpi=150,
    thread_count=4,
    use_pdftocairo=True,
)

mosaic.index_file(
    'hsfuhwfusgfhwi8',
    pdf_path,
    images=images,
    store_img_bs64=False
)