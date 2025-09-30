"""
Instance extraction algorithms

This module contains different algorithms for extracting instances from semantic segmentation masks.
"""

from .watershed import WatershedExtractor
from .connected_components import ConnectedComponentsExtractor

__all__ = [
    "WatershedExtractor",
    "ConnectedComponentsExtractor"
]
