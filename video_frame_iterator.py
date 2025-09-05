#!/usr/bin/env python3
"""
Video Frame Iterator with Metadata Collection

This module provides an iterator that processes video files frame by frame,
collecting metadata associated with each frame.
"""

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available. Install with: pip install numpy")

from typing import Iterator, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import json
import time
from datetime import datetime, timedelta


class VideoFrameMetadata:
    """Container for frame metadata"""
    
    def __init__(self, frame_number: int, timestamp: float, video_info: Dict[str, Any]):
        self.frame_number = frame_number
        self.timestamp = timestamp  # Timestamp in seconds
        self.video_info = video_info
        self.processing_time = None
        self.custom_metadata = {}
    
    def add_custom_metadata(self, key: str, value: Any):
        """Add custom metadata to this frame"""
        self.custom_metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            'frame_number': self.frame_number,
            'timestamp': self.timestamp,
            'timestamp_formatted': str(timedelta(seconds=self.timestamp)),
            'video_info': self.video_info,
            'processing_time': self.processing_time,
            'custom_metadata': self.custom_metadata
        }


class VideoFrameIterator:
    """
    Iterator that processes video files frame by frame with metadata collection
    
    Features:
    - Frame-by-frame iteration
    - Metadata collection for each frame
    - Support for various video formats
    - Optional frame skipping and sampling
    - Memory-efficient processing
    - Progress tracking
    """
    
    def __init__(
        self, 
        video_path: Union[str, Path],
        frame_skip: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        collect_frame_stats: bool = True,
        resize_frames: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize the video frame iterator
        
        Args:
            video_path: Path to the video file
            frame_skip: Skip every N frames (1 = process all frames)
            start_frame: Starting frame number
            end_frame: Ending frame number (None = process until end)
            collect_frame_stats: Whether to collect statistical metadata for each frame
            resize_frames: Optional tuple (width, height) to resize frames
        """
        self.video_path = Path(video_path)
        self.frame_skip = max(1, frame_skip)
        self.start_frame = max(0, start_frame)
        self.end_frame = end_frame
        self.collect_frame_stats = collect_frame_stats
        self.resize_frames = resize_frames
        
        # Check dependencies
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for video processing. Install with: pip install opencv-python")
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for video processing. Install with: pip install numpy")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Get video information
        self.video_info = self._get_video_info()
        
        # Set starting position
        if self.start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        
        self.current_frame_number = self.start_frame
        self.frames_processed = 0
        
    def _get_video_info(self) -> Dict[str, Any]:
        """Extract video metadata"""
        return {
            'filename': self.video_path.name,
            'path': str(self.video_path),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration_seconds': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) / self.cap.get(cv2.CAP_PROP_FPS),
            'fourcc': int(self.cap.get(cv2.CAP_PROP_FOURCC)),
            'codec': self._fourcc_to_string(int(self.cap.get(cv2.CAP_PROP_FOURCC)))
        }
    
    def _fourcc_to_string(self, fourcc: int) -> str:
        """Convert FOURCC code to string"""
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    def _collect_frame_statistics(self, frame) -> Dict[str, Any]:
        """Collect statistical metadata for a frame"""
        if not self.collect_frame_stats:
            return {}
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        return {
            'shape': frame.shape,
            'dtype': str(frame.dtype),
            'mean_brightness': float(np.mean(gray)),
            'std_brightness': float(np.std(gray)),
            'min_brightness': int(np.min(gray)),
            'max_brightness': int(np.max(gray)),
            'mean_rgb': [float(np.mean(frame[:,:,i])) for i in range(3)],
            'std_rgb': [float(np.std(frame[:,:,i])) for i in range(3)],
            'mean_hue': float(np.mean(hsv[:,:,0])),
            'mean_saturation': float(np.mean(hsv[:,:,1])),
            'mean_value': float(np.mean(hsv[:,:,2])),
            'file_size_bytes': frame.nbytes,
            'unique_colors': len(np.unique(frame.reshape(-1, frame.shape[-1]), axis=0))
        }
    
    def __iter__(self):
        """Make this object iterable"""
        return self
    
    def __next__(self):
        """
        Get the next frame with metadata
        
        Returns:
            Tuple of (frame_array, metadata_object)
        """
        start_time = time.time()
        
        # Check if we've reached the end
        if self.end_frame is not None and self.current_frame_number >= self.end_frame:
            self.cap.release()
            raise StopIteration
        
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        
        # Calculate timestamp
        timestamp = self.current_frame_number / self.video_info['fps']
        
        # Create metadata object
        metadata = VideoFrameMetadata(
            frame_number=self.current_frame_number,
            timestamp=timestamp,
            video_info=self.video_info
        )
        
        # Resize frame if requested
        if self.resize_frames:
            frame = cv2.resize(frame, self.resize_frames)
            metadata.add_custom_metadata('resized_to', self.resize_frames)
            metadata.add_custom_metadata('original_size', (self.video_info['width'], self.video_info['height']))
        
        # Collect frame statistics
        frame_stats = self._collect_frame_statistics(frame)
        for key, value in frame_stats.items():
            metadata.add_custom_metadata(key, value)
        
        # Record processing time
        metadata.processing_time = time.time() - start_time
        
        # Skip frames if needed
        for _ in range(self.frame_skip - 1):
            ret, _ = self.cap.read()
            if not ret:
                self.cap.release()
                raise StopIteration
            self.current_frame_number += 1
        
        self.current_frame_number += 1
        self.frames_processed += 1
        
        return frame, metadata
    
    def get_frame_at_time(self, timestamp_seconds: float):
        """
        Get a specific frame at a given timestamp
        
        Args:
            timestamp_seconds: Time in seconds to seek to
            
        Returns:
            Tuple of (frame_array, metadata_object)
        """
        frame_number = int(timestamp_seconds * self.video_info['fps'])
        return self.get_frame_at_number(frame_number)
    
    def get_frame_at_number(self, frame_number: int):
        """
        Get a specific frame by frame number
        
        Args:
            frame_number: Frame number to retrieve
            
        Returns:
            Tuple of (frame_array, metadata_object)
        """
        if frame_number < 0 or frame_number >= self.video_info['total_frames']:
            raise ValueError(f"Frame number {frame_number} is out of range [0, {self.video_info['total_frames']})")
        
        # Seek to the frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_number}")
        
        timestamp = frame_number / self.video_info['fps']
        
        metadata = VideoFrameMetadata(
            frame_number=frame_number,
            timestamp=timestamp,
            video_info=self.video_info
        )
        
        # Resize frame if requested
        if self.resize_frames:
            frame = cv2.resize(frame, self.resize_frames)
            metadata.add_custom_metadata('resized_to', self.resize_frames)
        
        # Collect frame statistics
        frame_stats = self._collect_frame_statistics(frame)
        for key, value in frame_stats.items():
            metadata.add_custom_metadata(key, value)
        
        return frame, metadata
    
    def save_metadata_batch(self, metadata_list: list, output_path: Union[str, Path]):
        """
        Save a batch of metadata to JSON file
        
        Args:
            metadata_list: List of VideoFrameMetadata objects
            output_path: Path to save the JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata_dicts = [metadata.to_dict() for metadata in metadata_list]
        
        with open(output_path, 'w') as f:
            json.dump(metadata_dicts, f, indent=2, default=str)
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information"""
        total_frames = self.video_info['total_frames']
        if self.end_frame:
            total_frames = min(total_frames, self.end_frame) - self.start_frame
        
        progress_percentage = (self.frames_processed * self.frame_skip) / total_frames * 100
        
        return {
            'current_frame': self.current_frame_number,
            'frames_processed': self.frames_processed,
            'total_frames': total_frames,
            'progress_percentage': progress_percentage,
            'estimated_remaining_frames': total_frames - (self.frames_processed * self.frame_skip)
        }
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        if self.cap:
            self.cap.release()
    
    def close(self):
        """Manually close the video capture"""
        if self.cap:
            self.cap.release()


class BatchVideoProcessor:
    """
    Process multiple videos and collect metadata in batches
    """
    
    def __init__(self, video_paths: list, **iterator_kwargs):
        """
        Initialize batch processor
        
        Args:
            video_paths: List of video file paths
            **iterator_kwargs: Arguments passed to VideoFrameIterator
        """
        self.video_paths = [Path(p) for p in video_paths]
        self.iterator_kwargs = iterator_kwargs
        self.all_metadata = []
    
    def process_all(self, metadata_callback: Optional[callable] = None) -> Dict[str, list]:
        """
        Process all videos and collect metadata
        
        Args:
            metadata_callback: Optional callback function called for each frame
                             Signature: callback(frame, metadata) -> modified_metadata
        
        Returns:
            Dictionary with video paths as keys and metadata lists as values
        """
        results = {}
        
        for video_path in self.video_paths:
            print(f"Processing video: {video_path}")
            video_metadata = []
            
            try:
                with VideoFrameIterator(video_path, **self.iterator_kwargs) as iterator:
                    for frame, metadata in iterator:
                        # Apply callback if provided
                        if metadata_callback:
                            metadata = metadata_callback(frame, metadata)
                        
                        video_metadata.append(metadata)
                        
                        # Print progress every 100 frames
                        if len(video_metadata) % 100 == 0:
                            progress = iterator.get_progress_info()
                            print(f"  Processed {progress['frames_processed']} frames "
                                  f"({progress['progress_percentage']:.1f}%)")
                
                results[str(video_path)] = video_metadata
                print(f"  Completed: {len(video_metadata)} frames processed")
                
            except Exception as e:
                print(f"  Error processing {video_path}: {e}")
                results[str(video_path)] = []
        
        return results
    
    def save_all_metadata(self, output_dir: Union[str, Path]):
        """Save metadata for all processed videos"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = self.process_all()
        
        for video_path, metadata_list in results.items():
            video_name = Path(video_path).stem
            output_file = output_dir / f"{video_name}_metadata.json"
            
            metadata_dicts = [metadata.to_dict() for metadata in metadata_list]
            with open(output_file, 'w') as f:
                json.dump(metadata_dicts, f, indent=2, default=str)
            
            print(f"Saved metadata for {video_name}: {output_file}")


# Example usage and testing functions
def example_metadata_callback(frame, metadata: VideoFrameMetadata) -> VideoFrameMetadata:
    """
    Example callback function that adds custom analysis to frame metadata
    """
    # Add motion detection
    if hasattr(example_metadata_callback, 'prev_frame'):
        diff = cv2.absdiff(frame, example_metadata_callback.prev_frame)
        motion_score = np.mean(diff)
        metadata.add_custom_metadata('motion_score', float(motion_score))
    
    example_metadata_callback.prev_frame = frame.copy()
    
    # Add edge detection score
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    metadata.add_custom_metadata('edge_density', float(edge_density))
    
    # Add color dominance
    colors = frame.reshape(-1, 3)
    dominant_color = np.mean(colors, axis=0)
    metadata.add_custom_metadata('dominant_color_bgr', dominant_color.tolist())
    
    return metadata


def demo_video_processing(video_path: str, output_dir: str = "video_metadata_output"):
    """
    Demonstration of video frame processing with metadata collection
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save metadata
    """
    print(f"Demo: Processing video {video_path}")
    
    # Example 1: Basic frame iteration with metadata
    print("\n=== Basic Frame Iteration ===")
    metadata_list = []
    
    try:
        with VideoFrameIterator(
            video_path, 
            frame_skip=10,  # Process every 10th frame
            collect_frame_stats=True,
            resize_frames=(640, 480)
        ) as iterator:
            
            for i, (frame, metadata) in enumerate(iterator):
                metadata_list.append(metadata)
                
                print(f"Frame {metadata.frame_number}: "
                      f"{metadata.timestamp:.2f}s, "
                      f"brightness={metadata.custom_metadata.get('mean_brightness', 0):.1f}")
                
                # Stop after processing 20 frames for demo
                if i >= 19:
                    break
        
        print(f"Processed {len(metadata_list)} frames")
        
        # Save metadata
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        iterator_instance = VideoFrameIterator(video_path)
        iterator_instance.save_metadata_batch(metadata_list, output_path / "demo_metadata.json")
        iterator_instance.close()
        
        print(f"Metadata saved to {output_path / 'demo_metadata.json'}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
    
    # Example 2: Batch processing with custom callback
    print("\n=== Batch Processing with Custom Analysis ===")
    try:
        processor = BatchVideoProcessor(
            [video_path],
            frame_skip=30,  # Process every 30th frame
            collect_frame_stats=True
        )
        
        results = processor.process_all(metadata_callback=example_metadata_callback)
        
        for video, metadata_list in results.items():
            print(f"Video: {video}")
            print(f"  Total frames processed: {len(metadata_list)}")
            if metadata_list:
                avg_motion = np.mean([m.custom_metadata.get('motion_score', 0) for m in metadata_list[1:]])
                avg_edge_density = np.mean([m.custom_metadata.get('edge_density', 0) for m in metadata_list])
                print(f"  Average motion score: {avg_motion:.2f}")
                print(f"  Average edge density: {avg_edge_density:.3f}")
        
    except Exception as e:
        print(f"Error during batch processing: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Frame Iterator with Metadata Collection")
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--output_dir', default='video_metadata_output', 
                       help='Output directory for metadata')
    parser.add_argument('--frame_skip', type=int, default=1, 
                       help='Process every Nth frame (default: 1)')
    parser.add_argument('--start_frame', type=int, default=0, 
                       help='Starting frame number')
    parser.add_argument('--end_frame', type=int, 
                       help='Ending frame number (optional)')
    parser.add_argument('--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                       help='Resize frames to specified dimensions')
    parser.add_argument('--demo', action='store_true', 
                       help='Run demonstration mode')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_video_processing(args.video_path, args.output_dir)
    else:
        # Basic usage
        resize_dims = tuple(args.resize) if args.resize else None
        
        print(f"Processing video: {args.video_path}")
        metadata_list = []
        
        try:
            with VideoFrameIterator(
                args.video_path,
                frame_skip=args.frame_skip,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                resize_frames=resize_dims
            ) as iterator:
                
                for frame, metadata in iterator:
                    metadata_list.append(metadata)
                    
                    if len(metadata_list) % 100 == 0:
                        progress = iterator.get_progress_info()
                        print(f"Progress: {progress['progress_percentage']:.1f}% "
                              f"({progress['frames_processed']} frames)")
            
            # Save results
            output_path = Path(args.output_dir)
            output_path.mkdir(exist_ok=True)
            
            video_name = Path(args.video_path).stem
            metadata_file = output_path / f"{video_name}_metadata.json"
            
            metadata_dicts = [metadata.to_dict() for metadata in metadata_list]
            with open(metadata_file, 'w') as f:
                json.dump(metadata_dicts, f, indent=2, default=str)
            
            print(f"\nCompleted! Processed {len(metadata_list)} frames")
            print(f"Metadata saved to: {metadata_file}")
            
        except Exception as e:
            print(f"Error: {e}")