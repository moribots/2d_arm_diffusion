"""
Utilities for recording and saving videos during policy validation.

This module provides reusable functions for:
- Saving frames as videos with various codec options
- Handling WandB video logging
- Handling fallback mechanisms when video saving fails
"""

import os
import time
import cv2
import numpy as np
import wandb
import traceback
from typing import List, Tuple, Optional, Union, Any
import imageio  # For GIF creation


def save_video(
    frames: List[np.ndarray], 
    base_path: str, 
    video_identifier: str = None, 
    fps: int = 30, 
    wandb_log: bool = True, 
    wandb_key: str = "validation_video",
    additional_wandb_data: dict = None,
    use_gif: bool = True  # Default to GIF output which has better compatibility
) -> Tuple[str, bool]:
    """
    Save a sequence of frames as a video file with robust fallback options.
    
    Args:
        frames: List of RGB numpy arrays representing video frames
        base_path: Base directory to save the video
        video_identifier: Unique identifier for the video (defaults to timestamp if None)
        fps: Frames per second for the video
        wandb_log: Whether to log the video to WandB
        wandb_key: Key to use when logging to WandB
        additional_wandb_data: Additional data to log to WandB alongside the video
        use_gif: Whether to save as GIF (more compatible) instead of MP4
        
    Returns:
        Tuple of (video_path, success) where:
            - video_path is the path to the saved video (or None if failed)
            - success is a boolean indicating if video was successfully saved
    """
    if len(frames) == 0:
        print("No frames to save")
        return None, False
        
    # Debug: Check first frame content
    first_frame = frames[0]
    print(f"First frame shape: {first_frame.shape}, dtype: {first_frame.dtype}")
    print(f"First frame min: {first_frame.min()}, max: {first_frame.max()}, mean: {first_frame.mean()}")
    
    # Ensure frames are in uint8 format with correct value range
    normalized_frames = []
    for i, frame in enumerate(frames):
        # If frame is float type, convert to uint8
        if frame.dtype != np.uint8:
            print(f"Converting frame {i} from {frame.dtype} to uint8")
            if frame.max() <= 1.0:  # Assuming 0-1 float range
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Ensure frame has 3 channels
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            print(f"Converting grayscale frame {i} to RGB")
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            print(f"Converting RGBA frame {i} to RGB")
        
        normalized_frames.append(frame)
    
    # Get frame dimensions
    height, width, _ = normalized_frames[0].shape
    
    # Create output directory
    video_dir = base_path
    os.makedirs(video_dir, exist_ok=True)
    
    # Generate a timestamp for unique identification if not provided
    if video_identifier is None:
        video_identifier = f"{int(time.time())}"
    
    # Debug: Check if the directory is writeable
    print(f"Attempting to save video to: {video_dir}")
    if not os.access(video_dir, os.W_OK):
        print(f"Warning: Directory {video_dir} is not writeable!")
    
    # Try saving individual frames as a fallback mechanism
    debug_frame_dir = os.path.join(video_dir, f"debug_frames_{video_identifier}")
    try:
        os.makedirs(debug_frame_dir, exist_ok=True)
        # Save first and last frame for debugging
        # Convert to BGR for OpenCV saving
        cv2.imwrite(os.path.join(debug_frame_dir, "first_frame.png"), 
                    cv2.cvtColor(normalized_frames[0], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(debug_frame_dir, "last_frame.png"), 
                    cv2.cvtColor(normalized_frames[-1], cv2.COLOR_RGB2BGR))
        print(f"Debug frames saved to {debug_frame_dir}")
    except Exception as e:
        print(f"Failed to save debug frames: {e}")
    
    # If GIF output is requested, use imageio
    if use_gif:
        try:
            gif_path = os.path.join(video_dir, f"validation_video_{video_identifier}.gif")
            
            print(f"Creating GIF with {len(normalized_frames)} frames at {fps} FPS")
            # Set a reasonable fps for GIFs (too high can make files enormous)
            gif_fps = min(fps, 20)  # Cap at 20 FPS for reasonable file size
            
            # May need to downsample frames to keep GIF size reasonable
            max_gif_width = 480
            if width > max_gif_width:
                scale_factor = max_gif_width / width
                new_width = max_gif_width
                new_height = int(height * scale_factor)
                resized_frames = []
                for frame in normalized_frames:
                    resized = cv2.resize(frame, (new_width, new_height), 
                                        interpolation=cv2.INTER_AREA)
                    resized_frames.append(resized)
                normalized_frames = resized_frames
                print(f"Resized frames to {new_width}x{new_height} for GIF")
            
            # Create GIF
            imageio.mimsave(gif_path, normalized_frames, fps=gif_fps)
            
            # Verify GIF was created successfully
            if os.path.exists(gif_path) and os.path.getsize(gif_path) > 0:
                print(f"GIF successfully saved to {gif_path} with size {os.path.getsize(gif_path)} bytes")
                
                # Log to WandB if requested
                if wandb_log and 'wandb' in globals() and wandb.run:
                    log_data = {wandb_key: wandb.Video(gif_path, format="gif", fps=gif_fps)}
                    if additional_wandb_data:
                        log_data.update(additional_wandb_data)
                    wandb.log(log_data)
                
                return gif_path, True
            else:
                print(f"Failed to save GIF or file is empty: {gif_path}")
                return None, False
                
        except Exception as e:
            print(f"Error creating GIF: {e}")
            print(traceback.format_exc())
            # Fall back to MP4 attempt
            print("Falling back to MP4 format...")
    
    # If we get here, either GIF creation failed or was not requested
    # Try MP4 format with various codecs
    try:
        # For OpenCV, we need to convert RGB frames to BGR
        bgr_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in normalized_frames]
        
        # List of codecs to try (prioritizing mp4 formats)
        codecs_to_try = [
            ('mp4v', '.mp4'),  # MP4 codec - most compatible with WandB
            ('avc1', '.mp4'),  # H.264 in MP4 container
            ('XVID', '.avi'),  # MPEG-4 in AVI container - fallback
            ('MJPG', '.avi'),  # Motion JPEG in AVI - very compatible
        ]
        
        video_writer = None
        saved_video_path = None
        
        for codec, ext in codecs_to_try:
            try:
                print(f"Trying codec: {codec} with extension {ext}")
                
                current_path = os.path.join(video_dir, f"validation_video_{video_identifier}{ext}")
                
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(current_path, fourcc, fps, (width, height))
                
                if writer.isOpened():
                    print(f"Successfully opened VideoWriter with codec {codec}")
                    video_writer = writer
                    video_path = current_path
                    saved_video_path = current_path
                    break
                else:
                    print(f"Could not open VideoWriter with codec {codec}")
                    writer.release()
            except Exception as e:
                print(f"Error with codec {codec}: {e}")
        
        if video_writer is None:
            print("Failed to initialize any VideoWriter. Falling back to image logging.")
            if wandb_log and 'wandb' in globals() and wandb.run:
                wandb.log({wandb_key + "_frames": [wandb.Image(frame) for frame in normalized_frames[:10]]})
            return None, False
            
        # At this point we have a working video_writer
        for frame in bgr_frames:  # Use BGR frames for OpenCV
            video_writer.write(frame)
        
        video_writer.release()
        print(f"Video writer released for {video_path}")
        
        # Verify the file exists and has content
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            print(f"Video successfully saved to {video_path} with size {os.path.getsize(video_path)} bytes")
            
            # Log the video and any additional data to WandB if requested
            if wandb_log and 'wandb' in globals() and wandb.run:
                log_data = {wandb_key: wandb.Video(video_path)}
                if additional_wandb_data:
                    log_data.update(additional_wandb_data)
                wandb.log(log_data)
                
            return video_path, True
        else:
            print(f"Failed to save video or file is empty: {video_path}")
            # Log frames as images instead as a fallback
            if wandb_log and 'wandb' in globals() and wandb.run:
                wandb.log({wandb_key + "_frames": [wandb.Image(frame) for frame in normalized_frames[:10]]})
            return None, False
            
    except Exception as e:
        print(f"Error saving video: {e}")
        print(traceback.format_exc())
        # Alternative: Save individual frames as images and log them
        if wandb_log and 'wandb' in globals() and wandb.run:
            wandb.log({wandb_key + "_frames": [wandb.Image(frame) for frame in normalized_frames[:10]]})
        return None, False
