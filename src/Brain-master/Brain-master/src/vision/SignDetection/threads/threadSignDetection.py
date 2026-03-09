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
import threading
import os

from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (
    mainCamera,
    SignDetection,
    SignPosition,
    SignConfidence,
    TrafficSignStats,
)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.allMessages import StateChange
from src.statemachine.systemMode import SystemMode

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class threadSignDetection(ThreadWithStop):
    """Thread which handles traffic sign detection using YOLOv8 model (traffic.pt).
    
    Detects traffic signs using:
    - YOLOv8 model trained on traffic signs (traffic.pt)
    - Real-time sign classification
    - Centroid-based sign tracking
    - False positive filtering
    
    Signs detected:
    - Speed limits (30, 50, 90 km/h)
    - Stop signs
    - Parking signs
    - Lane keep signs
    - Highway entry/exit signs
    - Traffic lights
    - Pedestrian crossings
    
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues.
        logging (logging object): Made for debugging.
        debugging (bool): A flag for debugging.
    """

    # ===== Model Configuration =====
    MODEL_PATH = "traffic.pt"  # Path to your YOLOv8 traffic signs model
    
    # ===== Sign Type Mapping =====
    # Map class IDs to readable names (adjust based on your model's classes)
    SIGN_CLASSES = {
        0: 'speed_limit_30',
        1: 'speed_limit_50',
        2: 'speed_limit_90',
        3: 'stop',
        4: 'parking',
        5: 'lane_keep',
        6: 'highway_entry',
        7: 'highway_exit',
        8: 'traffic_light',
        9: 'pedestrian_crossing'
    }

    # ================================ INIT ===============================================
    def __init__(self, queuesList, logger, debugger):
        super(threadSignDetection, self).__init__(pause=0.1)  # ~10 FPS
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        
        # ===== Image Parameters =====
        self.frame_width = 640
        self.frame_height = 480
        self.inference_size = (640, 640)  # YOLO input size
        
        # ===== YOLO Model Parameters =====
        self.model = None
        self.model_loaded = False
        self.model_loading_lock = threading.Lock()
        self.confidence_threshold = 0.4
        self.iou_threshold = 0.5
        
        # ===== Detection Filtering =====
        self.min_sign_area = 100  # pixels squared
        self.max_sign_area = 100000  # pixels squared
        
        # ===== Sign Tracking =====
        self.track_history_size = 5  # frames
        self.sign_tracker = {}  # {sign_id: deque of detections}
        self.track_id_counter = 0
        self.max_centroid_distance = 80  # pixels for centroid matching
        
        # ===== Persistent Detection Filtering =====
        self.sign_persistence_threshold = 2  # frames before confirming
        self.sign_confidence_history = {}  # {sign_id: [confidences]}
        
        # ===== Performance Tracking =====
        self.frame_count = 0
        self.inference_time = 0
        self.fps = 0
        self.last_time = time.time()
        self.processing_times = deque(maxlen=30)
        
        # Initialize message handlers
        self.subscribe()
        self._init_senders()
        
        # Load YOLO model
        self._load_yolo_model()

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
        """Initialize message senders for sign detection outputs."""
        self.signDetectionSender = messageHandlerSender(
            self.queuesList, SignDetection
        )
        self.signPositionSender = messageHandlerSender(
            self.queuesList, SignPosition
        )
        self.signConfidenceSender = messageHandlerSender(
            self.queuesList, SignConfidence
        )
        self.signStatsSender = messageHandlerSender(
            self.queuesList, TrafficSignStats
        )

    # ================================ LOAD YOLO MODEL =========================================
    def _load_yolo_model(self):
        """Load YOLOv8 traffic signs model (traffic.pt)."""
        try:
            if not YOLO_AVAILABLE:
                if self.debugger:
                    self.logger.error("YOLOv8 (ultralytics) not available. Install with: pip install ultralytics")
                self.model_loaded = False
                return
            
            if self.debugger:
                self.logger.info(f"Loading YOLOv8 traffic signs model from {self.MODEL_PATH}...")
            
            # Check if model file exists
            if not os.path.exists(self.MODEL_PATH):
                if self.debugger:
                    self.logger.error(f"Model file not found: {self.MODEL_PATH}")
                self.model_loaded = False
                return
            
            with self.model_loading_lock:
                # Load the traffic.pt model
                self.model = YOLO(self.MODEL_PATH)
                
                # Set to inference mode (CPU by default)
                self.model.to('cpu')  # Use 'cuda' if GPU available
                
                self.model_loaded = True
            
            if self.debugger:
                self.logger.info(f"YOLOv8 traffic signs model loaded successfully")
                
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Failed to load YOLOv8 model: {e}")
            self.model_loaded = False

    # ================================ RUN ================================================
    def thread_work(self):
        """Main thread work - detect signs from frame."""
        try:
            # Check if model is loaded
            if not self.model_loaded:
                return
            
            # Receive camera frame
            frame_data = self.cameraSubscriber.receive()
            if frame_data is None:
                return
            
            # Decode frame
            frame = self._decode_frame(frame_data)
            if frame is None:
                return
            
            # Perform sign detection
            start_time = time.time()
            signs = self._detect_signs(frame)
            inference_time = (time.time() - start_time) * 1000  # ms
            self.inference_time = inference_time
            
            # Track signs across frames
            if signs:
                tracked_signs = self._track_signs(signs)
                
                # Filter persistent detections
                filtered_signs = self._filter_persistent_signs(tracked_signs)
                
                # Send detection results
                self._send_sign_data(filtered_signs, inference_time)
            
            # Update FPS
            self._update_fps()
            self.frame_count += 1
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Sign Detection Error: {e}")

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

    # ================================ DETECT SIGNS =========================================
    def _detect_signs(self, frame):
        """Detect signs using YOLOv8 model."""
        try:
            signs = []
            
            with self.model_loading_lock:
                if not self.model_loaded:
                    return signs
                
                # Run YOLO inference
                results = self.model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
            
            # Extract detections
            if results and results[0].boxes:
                boxes = results[0].boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    
                    # Get class name
                    class_name = self.SIGN_CLASSES.get(class_id, f"unknown_{class_id}")
                    
                    # Calculate dimensions
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    area = w * h
                    
                    # Filter by area
                    if not (self.min_sign_area < area < self.max_sign_area):
                        continue
                    
                    # Create sign detection
                    signs.append({
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'area': area,
                        'centroid_x': x + w / 2,
                        'centroid_y': y + h / 2,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'sign_type': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'timestamp': time.time()
                    })
            
            return signs
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Sign detection error: {e}")
            return []

    # ================================ TRACK SIGNS =========================================
    def _track_signs(self, current_signs):
        """Simple centroid-based sign tracking."""
        try:
            tracked_signs = []
            used_sign_indices = set()
            
            # Try to match current signs with existing tracks
            for track_id, track_history in list(self.sign_tracker.items()):
                if len(track_history) == 0:
                    del self.sign_tracker[track_id]
                    continue
                
                last_sign = track_history[-1]
                last_centroid = (last_sign['centroid_x'], last_sign['centroid_y'])
                
                # Find closest sign
                best_match_idx = None
                best_distance = self.max_centroid_distance
                
                for idx, sign in enumerate(current_signs):
                    if idx in used_sign_indices:
                        continue
                    
                    current_centroid = (sign['centroid_x'], sign['centroid_y'])
                    distance = np.sqrt(
                        (last_centroid[0] - current_centroid[0])**2 +
                        (last_centroid[1] - current_centroid[1])**2
                    )
                    
                    # Prefer same sign type
                    if sign['class_name'] == last_sign['class_name'] and distance < best_distance:
                        best_distance = distance
                        best_match_idx = idx
                
                # If match found, update track
                if best_match_idx is not None:
                    used_sign_indices.add(best_match_idx)
                    self.sign_tracker[track_id].append(current_signs[best_match_idx])
                    tracked_signs.append({
                        'track_id': track_id,
                        'sign': current_signs[best_match_idx],
                        'track_length': len(self.sign_tracker[track_id])
                    })
                else:
                    # Track lost
                    if track_id in self.sign_tracker:
                        del self.sign_tracker[track_id]
                    if track_id in self.sign_confidence_history:
                        del self.sign_confidence_history[track_id]
            
            # Create new tracks for unmatched signs
            for idx, sign in enumerate(current_signs):
                if idx not in used_sign_indices:
                    track_id = self.track_id_counter
                    self.track_id_counter += 1
                    self.sign_tracker[track_id] = deque([sign], maxlen=self.track_history_size)
                    tracked_signs.append({
                        'track_id': track_id,
                        'sign': sign,
                        'track_length': 1
                    })
            
            return tracked_signs
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Sign tracking error: {e}")
            return []

    # ================================ FILTER PERSISTENT SIGNS =========================================
    def _filter_persistent_signs(self, tracked_signs):
        """Filter out false positives by requiring signs to appear in multiple frames."""
        try:
            filtered_signs = []
            
            for obj in tracked_signs:
                track_id = obj['track_id']
                sign = obj['sign']
                track_length = obj['track_length']
                confidence = sign['confidence']
                
                # Initialize confidence history if needed
                if track_id not in self.sign_confidence_history:
                    self.sign_confidence_history[track_id] = deque(maxlen=5)
                
                self.sign_confidence_history[track_id].append(confidence)
                
                # Calculate average confidence
                avg_confidence = np.mean(list(self.sign_confidence_history[track_id]))
                
                # Accept detection if:
                # 1. It has appeared in multiple frames, OR
                # 2. Average confidence is very high
                if track_length >= self.sign_persistence_threshold or avg_confidence > 0.75:
                    sign['avg_confidence'] = avg_confidence
                    filtered_signs.append(obj)
            
            return filtered_signs
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Persistence filtering error: {e}")
            return tracked_signs

    # ================================ SEND SIGN DATA =========================================
    def _send_sign_data(self, filtered_signs, inference_time):
        """Send sign detection results via message queue."""
        try:
            for obj in filtered_signs:
                sign = obj['sign']
                
                # Send main sign detection data
                self.signDetectionSender.send({
                    'track_id': obj['track_id'],
                    'sign_type': sign['sign_type'],
                    'class_name': sign['class_name'],
                    'confidence': sign['avg_confidence'] if 'avg_confidence' in sign else sign['confidence'],
                    'bbox': {
                        'x': sign['x'],
                        'y': sign['y'],
                        'w': sign['w'],
                        'h': sign['h'],
                        'x1': sign['x1'],
                        'y1': sign['y1'],
                        'x2': sign['x2'],
                        'y2': sign['y2'],
                        'area': sign['area']
                    },
                    'position': {
                        'centroid_x': sign['centroid_x'],
                        'centroid_y': sign['centroid_y'],
                        'horizontal': self._get_horizontal_position(sign['centroid_x']),
                        'vertical': self._get_vertical_position(sign['centroid_y'])
                    },
                    'timestamp': time.time()
                })
                
                # Send position data (for autonomous driving decisions)
                self.signPositionSender.send({
                    'track_id': obj['track_id'],
                    'class_name': sign['class_name'],
                    'x': sign['x'],
                    'y': sign['y'],
                    'centroid_x': sign['centroid_x'],
                    'centroid_y': sign['centroid_y']
                })
                
                # Send confidence level
                conf_value = sign['avg_confidence'] if 'avg_confidence' in sign else sign['confidence']
                self.signConfidenceSender.send(conf_value)
            
            # Send statistics every 30 frames
            if self.frame_count % 30 == 0:
                stats = {
                    'frame_count': self.frame_count,
                    'avg_inference_time': np.mean(list(self.processing_times)),
                    'fps': self.fps,
                    'active_tracks': len(self.sign_tracker),
                    'detections_count': len(filtered_signs),
                    'model_loaded': self.model_loaded,
                    'timestamp': time.time()
                }
                self.signStatsSender.send(stats)
                
                if self.debugger:
                    self.logger.info(
                        f"Sign Detection - FPS: {self.fps:.1f}, "
                        f"Detections: {len(filtered_signs)}, "
                        f"Active Tracks: {len(self.sign_tracker)}, "
                        f"Inference: {inference_time:.1f}ms"
                    )
                    
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Failed to send sign data: {e}")

    # ================================ POSITION HELPERS =========================================
    def _get_horizontal_position(self, centroid_x):
        """Determine horizontal position of sign."""
        frame_center = self.frame_width / 2
        offset = centroid_x - frame_center
        
        if abs(offset) < frame_center * 0.3:
            return 'center'
        elif offset < 0:
            return 'left'
        else:
            return 'right'
    
    def _get_vertical_position(self, centroid_y):
        """Determine vertical position of sign."""
        frame_middle = self.frame_height / 2
        offset = centroid_y - frame_middle
        
        if offset < -frame_middle * 0.3:
            return 'upper'
        elif offset > frame_middle * 0.3:
            return 'lower'
        else:
            return 'middle'

    # ================================ UPDATE FPS =========================================
    def _update_fps(self):
        """Calculate and update FPS."""
        try:
            current_time = time.time()
            frame_time = current_time - self.last_time
            
            if frame_time > 0:
                self.fps = 1.0 / frame_time
                self.processing_times.append(self.inference_time)
            
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
                    self.logger.info(f"Sign Detection mode changed to: {message}")
            except Exception as e:
                if self.debugger:
                    self.logger.error(f"State change error: {e}")

    # ================================ STOP ================================================
    def stop(self):
        """Clean up and stop the thread."""
        try:
            # Clean up YOLO model
            if self.model is not None:
                del self.model
                self.model = None
            
            # Clear tracking data
            self.sign_tracker.clear()
            self.sign_confidence_history.clear()
            
            if self.debugger:
                self.logger.info(
                    f"Sign Detection thread stopped. "
                    f"Processed {self.frame_count} frames, "
                    f"Avg FPS: {self.fps:.1f}"
                )
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Error during shutdown: {e}")
        
        super(threadSignDetection, self).stop()