from importlib.metadata import version

from .mosaic import Mosaic, QdrantClient

# __version__ = version("Mosaic")

__all__ = ["Mosaic", "QdrantClient"]