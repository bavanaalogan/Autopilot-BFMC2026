# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cv2
import numpy as np
import base64
import time
from collections import deque
from threading import Lock

from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (
    mainCamera,
    ProcessedFrame,
    FrameBuffer,
    FramePreprocessingStats,
    LightingCondition,
)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.allMessages import StateChange
from src.statemachine.systemMode import SystemMode


class threadFramePreprocessor(ThreadWithStop):
    """Thread which preprocesses camera frames for ML model inference.
    
    Preprocessing operations:
    - Frame resizing to model input size (640x480)
    - Pixel value normalization (0-1 or -1 to 1)
    - Histogram equalization for poor lighting conditions
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Brightness/contrast adjustment
    - Frame buffering (last 30 frames)
    - Motion detection and blur detection
    - Lighting condition analysis
    - Format conversion (BGR to RGB for models)
    
    Outputs preprocessed frames that are optimized for:
    - YOLOv8 object detection
    - CNN sign classification
    - Lane detection algorithms
    - Any other ML model inference
    
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues.
        logging (logging object): Made for debugging.
        debugging (bool): A flag for debugging.
    """

    # ===== Output Frame Sizes (adjust based on your models) =====
    OUTPUT_SIZES = {
        'yolo': (640, 640),           # YOLOv8 standard input
        'cnn': (224, 224),            # Generic CNN input
        'lane': (640, 480),           # Lane detection input
        'full': (640, 480),           # Full resolution
    }

    # ================================ INIT ===============================================
    def __init__(self, queuesList, logger, debugger):
        super(threadFramePreprocessor, self).__init__(pause=0.2)  # ~5 FPS (matches camera)
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        
        # ===== Input Frame Parameters =====
        self.input_width = 640
        self.input_height = 480
        
        # ===== Output Frame Parameters =====
        self.output_width = 640
        self.output_height = 480
        self.output_format = 'full'  # 'yolo', 'cnn', 'lane', or 'full'
        
        # ===== Normalization Method =====
        # Options: 'minmax' (0-1), 'zscore' (-1 to 1), 'imagenet' (ImageNet mean/std)
        self.normalization_method = 'minmax'
        
        # ===== Preprocessing Options =====
        self.enable_histogram_equalization = True
        self.enable_clahe = True  # Contrast Limited Adaptive Histogram Equalization
        self.enable_brightness_adjustment = True
        self.enable_motion_detection = True
        self.enable_blur_detection = True
        
        # ===== CLAHE Parameters =====
        self.clahe_clip_limit = 2.0
        self.clahe_tile_size = (8, 8)
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_size
        )
        
        # ===== Brightness Adjustment =====
        self.target_brightness = 127  # Target mean brightness (0-255)
        self.brightness_adjustment_strength = 0.5  # 0.0-1.0
        
        # ===== Motion Detection =====
        self.prev_frame = None
        self.motion_threshold = 5000  # Pixel count threshold for motion
        self.prev_frame_lock = Lock()
        
        # ===== Blur Detection =====
        self.blur_threshold = 100  # Laplacian variance threshold
        
        # ===== Frame Buffering =====
        self.frame_buffer_size = 30
        self.frame_buffer = deque(maxlen=self.frame_buffer_size)
        self.frame_buffer_lock = Lock()
        
        # ===== Lighting Conditions =====
        self.brightness_thresholds = {
            'very_dark': (0, 50),
            'dark': (50, 100),
            'low': (100, 150),
            'normal': (150, 200),
            'bright': (200, 230),
            'very_bright': (230, 255)
        }
        self.lighting_condition = 'normal'
        
        # ===== Performance Tracking =====
        self.frame_count = 0
        self.preprocessing_time = 0
        self.fps = 0
        self.last_time = time.time()
        self.blur_count = 0
        self.motion_count = 0
        
        # Initialize message handlers
        self.subscribe()
        self._init_senders()

    # ================================ SUBSCRIBE ===============================================
    def subscribe(self):
        """Subscribe to camera frames and state changes."""
        self.cameraSubscriber = messageHandlerSubscriber(
            self.queuesList, mainCamera, "lastOnly", True
        )
        self.stateChangeSubscriber = messageHandlerSubscriber(
            self.queuesList, StateChange, "lastOnly", True
        )

    # ================================ SENDERS ===============================================
    def _init_senders(self):
        """Initialize message senders for preprocessed frames."""
        self.processedFrameSender = messageHandlerSender(
            self.queuesList, ProcessedFrame
        )
        self.frameBufferSender = messageHandlerSender(
            self.queuesList, FrameBuffer
        )
        self.preprocessingStatsSender = messageHandlerSender(
            self.queuesList, FramePreprocessingStats
        )
        self.lightingConditionSender = messageHandlerSender(
            self.queuesList, LightingCondition
        )

    # ================================ RUN ================================================
    def thread_work(self):
        """Main thread work - preprocess frames."""
        try:
            # Receive camera frame
            frame_data = self.cameraSubscriber.receive()
            if frame_data is None:
                return
            
            # Decode frame
            frame = self._decode_frame(frame_data)
            if frame is None:
                return
            
            # Perform preprocessing
            start_time = time.time()
            processed_frame = self._preprocess_frame(frame)
            preprocessing_time = (time.time() - start_time) * 1000  # ms
            self.preprocessing_time = preprocessing_time
            
            if processed_frame is not None:
                # Store in buffer
                self._add_to_buffer(processed_frame)
                
                # Send processed frame
                self._send_processed_frame(processed_frame)
                
                # Send statistics periodically
                if self.frame_count % 30 == 0:
                    self._send_statistics()
            
            # Update FPS
            self._update_fps()
            self.frame_count += 1
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Frame Preprocessing Error: {e}")

    # ================================ DECODE FRAME =========================================
    def _decode_frame(self, frame_data):
        """Decode base64 encoded frame from camera."""
        try:
            if isinstance(frame_data, str):
                image_data = base64.b64decode(frame_data)
                frame = np.frombuffer(image_data, dtype=np.uint8)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                return frame
            return None
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Frame decode error: {e}")
            return None

    # ================================ PREPROCESS FRAME =========================================
    def _preprocess_frame(self, frame):
        """Perform all preprocessing operations on frame."""
        try:
            # Step 1: Analyze lighting condition
            self.lighting_condition = self._analyze_lighting(frame)
            
            # Step 2: Check for blur
            is_blurry = self._detect_blur(frame)
            if is_blurry:
                self.blur_count += 1
                if self.debugger and self.frame_count % 100 == 0:
                    self.logger.warning(f"Blurry frame detected (blur count: {self.blur_count})")
            
            # Step 3: Detect motion
            has_motion = self._detect_motion(frame)
            if has_motion:
                self.motion_count += 1
            
            # Step 4: Adjust brightness if needed
            if self.enable_brightness_adjustment:
                frame = self._adjust_brightness(frame)
            
            # Step 5: Apply histogram equalization
            if self.enable_histogram_equalization or self.lighting_condition in ['dark', 'very_dark', 'low']:
                frame = self._apply_histogram_equalization(frame)
            
            # Step 6: Apply CLAHE for better local contrast
            if self.enable_clahe:
                frame = self._apply_clahe(frame)
            
            # Step 7: Resize to output size
            frame = self._resize_frame(frame)
            
            # Step 8: Convert BGR to RGB (standard for ML models)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Step 9: Normalize pixel values
            frame = self._normalize_frame(frame)
            
            # Create output dictionary
            processed_frame = {
                'frame': frame,
                'original_frame': frame,  # Also include original
                'width': frame.shape[1],
                'height': frame.shape[0],
                'channels': frame.shape[2] if len(frame.shape) > 2 else 1,
                'lighting_condition': self.lighting_condition,
                'is_blurry': is_blurry,
                'has_motion': has_motion,
                'preprocessing_time': self.preprocessing_time,
                'frame_count': self.frame_count,
                'timestamp': time.time()
            }
            
            return processed_frame
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Preprocessing error: {e}")
            return None

    # ================================ ANALYZE LIGHTING =========================================
    def _analyze_lighting(self, frame):
        """Analyze lighting conditions of frame."""
        try:
            # Convert to grayscale for brightness analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            # Determine lighting condition
            for condition, (lower, upper) in self.brightness_thresholds.items():
                if lower <= mean_brightness < upper:
                    return condition
            
            return 'very_bright'  # Default for very bright images
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Lighting analysis error: {e}")
            return 'normal'

    # ================================ DETECT BLUR =========================================
    def _detect_blur(self, frame):
        """Detect if frame is blurry using Laplacian variance."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # If variance is low, frame is blurry
            is_blurry = laplacian_var < self.blur_threshold
            
            return is_blurry
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Blur detection error: {e}")
            return False

    # ================================ DETECT MOTION =========================================
    def _detect_motion(self, frame):
        """Detect motion between current and previous frame."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            has_motion = False
            
            with self.prev_frame_lock:
                if self.prev_frame is not None:
                    # Calculate absolute difference
                    diff = cv2.absdiff(self.prev_frame, gray)
                    
                    # Count pixels with significant change
                    motion_pixels = np.count_nonzero(diff > 30)
                    
                    has_motion = motion_pixels > self.motion_threshold
                
                # Update previous frame
                self.prev_frame = gray.copy()
            
            return has_motion
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Motion detection error: {e}")
            return False

    # ================================ ADJUST BRIGHTNESS =========================================
    def _adjust_brightness(self, frame):
        """Adjust brightness to target level."""
        try:
            # Convert to LAB color space for brightness adjustment
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel = lab[:,:,0]
            
            # Calculate current brightness
            current_brightness = np.mean(l_channel)
            
            # Calculate adjustment
            brightness_diff = self.target_brightness - current_brightness
            adjustment = brightness_diff * self.brightness_adjustment_strength
            
            # Apply adjustment (only if needed)
            if abs(adjustment) > 5:
                l_channel = np.clip(l_channel.astype(np.float32) + adjustment, 0, 255).astype(np.uint8)
                lab[:,:,0] = l_channel
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return frame
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Brightness adjustment error: {e}")
            return frame

    # ================================ HISTOGRAM EQUALIZATION =========================================
    def _apply_histogram_equalization(self, frame):
        """Apply histogram equalization for poor lighting."""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Equalize V (value/brightness) channel
            hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
            
            # Convert back to BGR
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            return frame
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Histogram equalization error: {e}")
            return frame

    # ================================ APPLY CLAHE =========================================
    def _apply_clahe(self, frame):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        try:
            # Convert to LAB
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel = lab[:,:,0]
            
            # Apply CLAHE to L channel
            l_channel = self.clahe.apply(l_channel)
            
            # Recombine
            lab[:,:,0] = l_channel
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return frame
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"CLAHE error: {e}")
            return frame

    # ================================ RESIZE FRAME =========================================
    def _resize_frame(self, frame):
        """Resize frame to output size."""
        try:
            output_size = self.OUTPUT_SIZES.get(self.output_format, (640, 480))
            
            # Resize with interpolation
            frame = cv2.resize(
                frame,
                output_size,
                interpolation=cv2.INTER_LINEAR
            )
            
            return frame
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Resize error: {e}")
            return frame

    # ================================ NORMALIZE FRAME =========================================
    def _normalize_frame(self, frame):
        """Normalize pixel values based on normalization method."""
        try:
            frame = frame.astype(np.float32)
            
            if self.normalization_method == 'minmax':
                # Normalize to 0-1
                frame = frame / 255.0
            
            elif self.normalization_method == 'zscore':
                # Normalize to -1 to 1 (zero-mean, unit variance)
                mean = np.mean(frame)
                std = np.std(frame)
                if std > 0:
                    frame = (frame - mean) / std
                frame = np.clip(frame, -1, 1)
            
            elif self.normalization_method == 'imagenet':
                # ImageNet normalization
                imagenet_mean = np.array([0.485, 0.456, 0.406]) * 255
                imagenet_std = np.array([0.229, 0.224, 0.225]) * 255
                
                frame = (frame - imagenet_mean) / imagenet_std
            
            return frame
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Normalization error: {e}")
            return frame / 255.0  # Default to minmax

    # ================================ ADD TO BUFFER =========================================
    def _add_to_buffer(self, processed_frame):
        """Add processed frame to circular buffer."""
        try:
            with self.frame_buffer_lock:
                # Encode frame as base64 for storage
                frame_uint8 = (processed_frame['frame'] * 255).astype(np.uint8)
                _, encoded = cv2.imencode('.jpg', frame_uint8)
                frame_base64 = base64.b64encode(encoded).decode('utf-8')
                
                # Create buffer entry
                buffer_entry = {
                    'frame_base64': frame_base64,
                    'width': processed_frame['width'],
                    'height': processed_frame['height'],
                    'lighting_condition': processed_frame['lighting_condition'],
                    'is_blurry': processed_frame['is_blurry'],
                    'has_motion': processed_frame['has_motion'],
                    'timestamp': processed_frame['timestamp']
                }
                
                self.frame_buffer.append(buffer_entry)
                
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Buffer add error: {e}")

    # ================================ SEND PROCESSED FRAME =========================================
    def _send_processed_frame(self, processed_frame):
        """Send processed frame via message queue."""
        try:
            # Encode frame as base64
            frame_uint8 = (processed_frame['frame'] * 255).astype(np.uint8)
            _, encoded = cv2.imencode('.jpg', frame_uint8)
            frame_base64 = base64.b64encode(encoded).decode('utf-8')
            
            # Send processed frame
            self.processedFrameSender.send(frame_base64)
            
            # Send lighting condition
            self.lightingConditionSender.send(self.lighting_condition)
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Failed to send processed frame: {e}")

    # ================================ SEND STATISTICS =========================================
    def _send_statistics(self):
        """Send preprocessing statistics."""
        try:
            with self.frame_buffer_lock:
                buffer_size = len(self.frame_buffer)
            
            stats = {
                'frame_count': self.frame_count,
                'preprocessing_time_ms': self.preprocessing_time,
                'fps': self.fps,
                'buffer_size': buffer_size,
                'buffer_capacity': self.frame_buffer_size,
                'blur_count': self.blur_count,
                'motion_count': self.motion_count,
                'lighting_condition': self.lighting_condition,
                'blur_percentage': (self.blur_count / max(self.frame_count, 1)) * 100,
                'motion_percentage': (self.motion_count / max(self.frame_count, 1)) * 100,
                'timestamp': time.time()
            }
            
            self.preprocessingStatsSender.send(stats)
            
            if self.debugger:
                self.logger.info(
                    f"Frame Preprocessing - FPS: {self.fps:.1f}, "
                    f"Time: {self.preprocessing_time:.1f}ms, "
                    f"Buffer: {buffer_size}/{self.frame_buffer_size}, "
                    f"Lighting: {self.lighting_condition}, "
                    f"Blur: {self.blur_percentage:.1f}%, "
                    f"Motion: {self.motion_percentage:.1f}%"
                )
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Failed to send statistics: {e}")

    # ================================ UPDATE FPS =========================================
    def _update_fps(self):
        """Calculate and update FPS."""
        try:
            current_time = time.time()
            frame_time = current_time - self.last_time
            
            if frame_time > 0:
                self.fps = 1.0 / frame_time
            
            self.last_time = current_time
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"FPS update error: {e}")

    # ================================ STATE CHANGE HANDLER ========================================
    def state_change_handler(self):
        """Handle state changes from the state machine."""
        message = self.stateChangeSubscriber.receive()
        if message is not None:
            try:
                modeDict = SystemMode[message].value["vision"]["thread"]
                
                if self.debugger:
                    self.logger.info(f"Frame Preprocessor mode changed to: {message}")
            except Exception as e:
                if self.debugger:
                    self.logger.error(f"State change error: {e}")

    # ================================ STOP ================================================
    def stop(self):
        """Clean up and stop the thread."""
        try:
            # Clear buffers
            with self.frame_buffer_lock:
                self.frame_buffer.clear()
            
            with self.prev_frame_lock:
                self.prev_frame = None
            
            if self.debugger:
                blur_rate = (self.blur_count / max(self.frame_count, 1)) * 100
                motion_rate = (self.motion_count / max(self.frame_count, 1)) * 100
                
                self.logger.info(
                    f"Frame Preprocessor thread stopped. "
                    f"Processed {self.frame_count} frames, "
                    f"Avg FPS: {self.fps:.1f}, "
                    f"Blur Rate: {blur_rate:.1f}%, "
                    f"Motion Rate: {motion_rate:.1f}%"
                )
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Error during shutdown: {e}")
        
        super(threadFramePreprocessor, self).stop()