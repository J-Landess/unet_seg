"""
Video Processing Module

A dedicated module for video frame processing with tensor support for deep learning applications.
"""

from .frame_iterator import VideoFrameIterator, VideoFrameMetadata, TensorFrameBatcher, BatchVideoProcessor

__all__ = [
    'VideoFrameIterator',
    'VideoFrameMetadata', 
    'TensorFrameBatcher',
    'BatchVideoProcessor'
]

__version__ = "1.0.0"
