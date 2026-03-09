import cv2
import numpy as np
import base64
import time
from collections import deque
from ultralytics import YOLO
import threading

from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (
    mainCamera,
    ObjectDetection,
    ObstacleWarning,
    DetectionFPS,
    DetectionStats,
)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.allMessages import StateChange
from src.statemachine.systemMode import SystemMode


class threadObjectDetection(ThreadWithStop):
    """Thread which will handle object detection using YOLOv8.
    
    Detects objects in real-time using:
    - YOLOv8 model for neural network inference
    - Multi-class object detection (cars, people, bicycles, etc.)
    - Distance estimation from bounding box size
    - False positive filtering
    - Object tracking across frames
    
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool): A flag for debugging.
    """

    # ===== Detected Object Classes (YOLO80 classes) =====
    YOLO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'cat', 15: 'dog', 16: 'horse', 17: 'sheep', 18: 'cow', 19: 'elephant',
        20: 'bear', 21: 'zebra', 22: 'giraffe', 23: 'backpack', 24: 'umbrella',
        25: 'handbag', 26: 'tie', 27: 'suitcase', 28: 'frisbee', 29: 'skis',
        30: 'snowboard', 31: 'sports ball', 32: 'kite', 33: 'baseball bat',
        34: 'baseball glove', 35: 'skateboard', 36: 'surfboard', 37: 'tennis racket',
        38: 'bottle', 39: 'wine glass', 40: 'cup', 41: 'fork', 42: 'knife',
        43: 'spoon', 44: 'bowl', 45: 'banana', 46: 'apple', 47: 'sandwich',
        48: 'orange', 49: 'broccoli', 50: 'carrot', 51: 'hot dog', 52: 'pizza',
        53: 'donut', 54: 'cake', 55: 'chair', 56: 'couch', 57: 'potted plant',
        58: 'bed', 59: 'dining table', 60: 'toilet', 61: 'tv', 62: 'laptop',
        63: 'mouse', 64: 'remote', 65: 'keyboard', 66: 'microwave', 67: 'oven',
        68: 'toaster', 69: 'sink', 70: 'refrigerator', 71: 'book', 72: 'clock',
        73: 'vase', 74: 'scissors', 75: 'teddy bear', 76: 'hair drier',
        77: 'toothbrush', 78: 'traffic sign', 79: 'pedestrian crossing'
    }
    
    # ===== Critical Classes for Autonomous Driving =====
    CRITICAL_CLASSES = {
        'person': 0,
        'bicycle': 1,
        'car': 2,
        'motorcycle': 3,
        'bus': 5,
        'truck': 7,
        'stop sign': 11,
        'traffic light': 9,
        'pedestrian crossing': 79
    }

    # ================================ INIT ===============================================
    def __init__(self, queuesList, logger, debugger):
        super(threadObjectDetection, self).__init__(pause=0.067)  # ~15 FPS
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        
        # ===== YOLO Model Parameters =====
        self.model_name = "yolov8m"  # yolov8n (nano), yolov8s (small), yolov8m (medium), yolov8l (large)
        self.model = None
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        self.model_loading_lock = threading.Lock()
        self.model_loaded = False
        
        # ===== Image Parameters =====
        self.frame_width = 640
        self.frame_height = 480
        self.inference_size = (640, 640)  # YOLO input size
        
        # ===== Detection Filtering =====
        self.min_detection_area = 100  # pixels squared
        self.max_detection_area = 307200  # 640x480 pixels
        
        # ===== Distance Estimation (Camera Calibration) =====
        # These need to be calibrated for your specific camera
        self.focal_length_pixels = 600  # pixels (camera calibration parameter)
        self.known_car_width = 1.8  # meters (real world)
        self.known_person_height = 1.7  # meters (real world)
        self.min_distance_threshold = 0.5  # meters (minimum detectable distance)
        self.max_distance_threshold = 50.0  # meters (maximum detectable distance)
        
        # ===== Collision Detection =====
        self.collision_distance_threshold = 3.0  # meters
        self.collision_confidence_threshold = 0.7
        self.collision_area_threshold = 15000  # pixels squared
        
        # ===== Object Tracking =====
        self.track_history_size = 10  # frames to track
        self.object_tracker = {}  # {object_id: deque of detections}
        self.track_id_counter = 0
        self.max_centroid_distance = 50  # pixels for centroid matching
        
        # ===== Persistent False Positive Filtering =====
        self.object_persistence_threshold = 3  # frames before confirming object
        self.object_confidence_history = {}  # {object_id: [confidences]}
        
        # ===== Performance Tracking =====
        self.frame_count = 0
        self.inference_time = 0
        self.fps = 0
        self.last_time = time.time()
        self.processing_times = deque(maxlen=30)  # Last 30 frames
        
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
        """Initialize message senders for object detection outputs."""
        self.objectDetectionSender = messageHandlerSender(
            self.queuesList, ObjectDetection
        )
        self.obstacleWarningSender = messageHandlerSender(
            self.queuesList, ObstacleWarning
        )
        self.detectionFPSSender = messageHandlerSender(
            self.queuesList, DetectionFPS
        )
        self.detectionStatsSender = messageHandlerSender(
            self.queuesList, DetectionStats
        )

    # ================================ LOAD YOLO MODEL =========================================
    def _load_yolo_model(self):
        """Load YOLOv8 model in a thread-safe manner."""
        try:
            if self.debugger:
                self.logger.info(f"Loading YOLOv8 model ({self.model_name})...")
            
            with self.model_loading_lock:
                # Load YOLOv8 model
                self.model = YOLO(f"{self.model_name}.pt")
                
                # Set inference device
                self.model.to('cpu')  # Use 'cuda' if GPU available
                
                self.model_loaded = True
                
            if self.debugger:
                self.logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Failed to load YOLO model: {e}")
            self.model_loaded = False

    # ================================ RUN ================================================
    def thread_work(self):
        """Main thread work - capture frame and perform object detection."""
        try:
            # Check if model is loaded
            if not self.model_loaded:
                return
            
            # Receive camera frame
            frame_data = self.cameraSubscriber.receive()
            if frame_data is None:
                return
            
            # Decode frame from base64
            frame = self._decode_frame(frame_data)
            if frame is None:
                return
            
            # Perform object detection
            start_time = time.time()
            detections = self._detect_objects(frame)
            inference_time = (time.time() - start_time) * 1000  # ms
            self.inference_time = inference_time
            
            # Track objects across frames
            if detections:
                tracked_objects = self._track_objects(detections)
                
                # Filter persistent detections
                filtered_objects = self._filter_persistent_detections(tracked_objects)
                
                # Estimate distances
                detection_data = self._estimate_distances(filtered_objects, frame)
                
                # Check for collision threats
                warnings = self._check_collisions(detection_data)
                
                # Send detection results
                self._send_detection_data(detection_data, warnings, inference_time)
            
            # Update FPS
            self._update_fps()
            self.frame_count += 1
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Object Detection Error: {e}")

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

    # ================================ DETECT OBJECTS =========================================
    def _detect_objects(self, frame):
        """Run YOLO inference on frame."""
        try:
            with self.model_loading_lock:
                if not self.model_loaded:
                    return []
                
                # Run inference
                results = self.model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                # Extract detections
                detections = []
                if results and results[0].boxes:
                    boxes = results[0].boxes
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()
                        class_id = int(box.cls[0].item())
                        class_name = self.YOLO_CLASSES.get(class_id, "unknown")
                        
                        # Basic filtering
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        if self.min_detection_area < area < self.max_detection_area:
                            detections.append({
                                'class_id': class_id,
                                'class_name': class_name,
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'width': width,
                                'height': height,
                                'area': area,
                                'confidence': confidence,
                                'centroid_x': (x1 + x2) / 2,
                                'centroid_y': (y1 + y2) / 2,
                                'timestamp': time.time()
                            })
                
                return detections
                
        except Exception as e:
            if self.debugger:
                self.logger.error(f"YOLO inference error: {e}")
            return []

    # ================================ TRACK OBJECTS =========================================
    def _track_objects(self, current_detections):
        """Simple centroid-based object tracking."""
        try:
            tracked_objects = []
            used_detection_indices = set()
            
            # Try to match current detections with existing tracks
            for track_id, track_history in list(self.object_tracker.items()):
                if len(track_history) == 0:
                    del self.object_tracker[track_id]
                    continue
                
                last_detection = track_history[-1]
                last_centroid = (last_detection['centroid_x'], last_detection['centroid_y'])
                
                # Find closest detection
                best_match_idx = None
                best_distance = self.max_centroid_distance
                
                for idx, detection in enumerate(current_detections):
                    if idx in used_detection_indices:
                        continue
                    
                    current_centroid = (detection['centroid_x'], detection['centroid_y'])
                    distance = np.sqrt(
                        (last_centroid[0] - current_centroid[0])**2 +
                        (last_centroid[1] - current_centroid[1])**2
                    )
                    
                    if distance < best_distance and detection['class_name'] == last_detection['class_name']:
                        best_distance = distance
                        best_match_idx = idx
                
                # If match found, update track
                if best_match_idx is not None:
                    used_detection_indices.add(best_match_idx)
                    self.object_tracker[track_id].append(current_detections[best_match_idx])
                    tracked_objects.append({
                        'track_id': track_id,
                        'detection': current_detections[best_match_idx],
                        'track_length': len(self.object_tracker[track_id])
                    })
                else:
                    # Track lost
                    if len(self.object_tracker[track_id]) > 0:
                        del self.object_tracker[track_id]
                    if track_id in self.object_confidence_history:
                        del self.object_confidence_history[track_id]
            
            # Create new tracks for unmatched detections
            for idx, detection in enumerate(current_detections):
                if idx not in used_detection_indices:
                    track_id = self.track_id_counter
                    self.track_id_counter += 1
                    self.object_tracker[track_id] = deque([detection], maxlen=self.track_history_size)
                    tracked_objects.append({
                        'track_id': track_id,
                        'detection': detection,
                        'track_length': 1
                    })
            
            return tracked_objects
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Object tracking error: {e}")
            return []

    # ================================ FILTER PERSISTENT DETECTIONS =========================================
    def _filter_persistent_detections(self, tracked_objects):
        """Filter out false positives by requiring objects to appear in multiple frames."""
        try:
            filtered_objects = []
            
            for obj in tracked_objects:
                track_id = obj['track_id']
                detection = obj['detection']
                track_length = obj['track_length']
                confidence = detection['confidence']
                
                # Initialize confidence history if needed
                if track_id not in self.object_confidence_history:
                    self.object_confidence_history[track_id] = deque(maxlen=5)
                
                self.object_confidence_history[track_id].append(confidence)
                
                # Calculate average confidence
                avg_confidence = np.mean(list(self.object_confidence_history[track_id]))
                
                # Accept detection if:
                # 1. It has appeared in multiple frames
                # 2. Average confidence is above threshold
                if track_length >= self.object_persistence_threshold or avg_confidence > 0.8:
                    detection['avg_confidence'] = avg_confidence
                    filtered_objects.append(obj)
            
            return filtered_objects
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Persistence filtering error: {e}")
            return tracked_objects

    # ================================ ESTIMATE DISTANCES =========================================
    def _estimate_distances(self, tracked_objects, frame):
        """Estimate distance to detected objects using size heuristics."""
        try:
            detection_data = []
            
            for obj in tracked_objects:
                detection = obj['detection']
                
                # Estimate distance based on bounding box size
                distance = self._calculate_distance(
                    detection['class_name'],
                    detection['width'],
                    detection['height']
                )
                
                # Estimate horizontal position (left, center, right)
                frame_center = self.frame_width / 2
                centroid_offset = detection['centroid_x'] - frame_center
                
                if abs(centroid_offset) < frame_center * 0.3:
                    horizontal_position = 'center'
                elif centroid_offset < 0:
                    horizontal_position = 'left'
                else:
                    horizontal_position = 'right'
                
                # Estimate vertical position (upper, middle, lower)
                frame_middle = self.frame_height / 2
                vertical_offset = detection['centroid_y'] - frame_middle
                
                if vertical_offset < -frame_middle * 0.3:
                    vertical_position = 'upper'
                elif vertical_offset > frame_middle * 0.3:
                    vertical_position = 'lower'
                else:
                    vertical_position = 'middle'
                
                detection_data.append({
                    'track_id': obj['track_id'],
                    'class_name': detection['class_name'],
                    'class_id': detection['class_id'],
                    'confidence': detection['avg_confidence'] if 'avg_confidence' in detection else detection['confidence'],
                    'distance': distance,
                    'bbox': {
                        'x1': detection['x1'],
                        'y1': detection['y1'],
                        'x2': detection['x2'],
                        'y2': detection['y2'],
                        'width': detection['width'],
                        'height': detection['height'],
                        'area': detection['area']
                    },
                    'position': {
                        'horizontal': horizontal_position,
                        'vertical': vertical_position,
                        'centroid_x': detection['centroid_x'],
                        'centroid_y': detection['centroid_y']
                    },
                    'timestamp': time.time()
                })
            
            return detection_data
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Distance estimation error: {e}")
            return []

    # ================================ CALCULATE DISTANCE =========================================
    def _calculate_distance(self, class_name, bbox_width, bbox_height):
        """Estimate distance using focal length and known object sizes."""
        try:
            # Use height for distance calculation (more stable)
            if class_name == 'person':
                real_height = self.known_person_height
                distance = (real_height * self.focal_length_pixels) / bbox_height
            elif class_name in ['car', 'bus', 'truck']:
                real_width = self.known_car_width
                distance = (real_width * self.focal_length_pixels) / bbox_width
            else:
                # Fallback: estimate based on area
                real_size = 1.0  # 1 meter
                area = bbox_width * bbox_height
                distance = (real_size * self.focal_length_pixels) / np.sqrt(area)
            
            # Clamp distance to reasonable range
            distance = np.clip(
                distance,
                self.min_distance_threshold,
                self.max_distance_threshold
            )
            
            return distance
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Distance calculation error: {e}")
            return self.max_distance_threshold

    # ================================ CHECK COLLISIONS =========================================
    def _check_collisions(self, detection_data):
        """Identify collision threats."""
        warnings = []
        
        try:
            for detection in detection_data:
                is_threat = False
                threat_level = 'low'
                
                # Check if object is critical and close
                if detection['class_name'] in self.CRITICAL_CLASSES:
                    distance = detection['distance']
                    confidence = detection['confidence']
                    area = detection['bbox']['area']
                    
                    # Calculate collision risk
                    if distance < self.collision_distance_threshold:
                        if confidence > self.collision_confidence_threshold:
                            is_threat = True
                            
                            if distance < 1.0:
                                threat_level = 'critical'
                            elif distance < 2.0:
                                threat_level = 'high'
                            else:
                                threat_level = 'medium'
                    
                    # Large close objects are more threatening
                    if area > self.collision_area_threshold and distance < 5.0:
                        is_threat = True
                        if threat_level == 'low':
                            threat_level = 'high'
                
                if is_threat:
                    warnings.append({
                        'track_id': detection['track_id'],
                        'class_name': detection['class_name'],
                        'distance': detection['distance'],
                        'threat_level': threat_level,
                        'position': detection['position'],
                        'confidence': detection['confidence'],
                        'timestamp': time.time()
                    })
        
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Collision detection error: {e}")
        
        return warnings

    # ================================ SEND DETECTION DATA =========================================
    def _send_detection_data(self, detection_data, warnings, inference_time):
        """Send object detection results via message queue."""
        try:
            # Send main detection data
            if detection_data:
                self.objectDetectionSender.send({
                    'detections': detection_data,
                    'timestamp': time.time(),
                    'frame_count': self.frame_count,
                    'inference_time_ms': inference_time
                })
            
            # Send collision warnings
            if warnings:
                for warning in warnings:
                    self.obstacleWarningSender.send(warning)
            
            # Send FPS information
            self.detectionFPSSender.send(self.fps)
            
            # Send statistics every 30 frames
            if self.frame_count % 30 == 0:
                stats = {
                    'frame_count': self.frame_count,
                    'avg_inference_time': np.mean(list(self.processing_times)),
                    'fps': self.fps,
                    'active_tracks': len(self.object_tracker),
                    'detections_count': len(detection_data),
                    'warnings_count': len(warnings)
                }
                self.detectionStatsSender.send(stats)
                
                if self.debugger:
                    self.logger.info(
                        f"Object Detection - FPS: {self.fps:.1f}, "
                        f"Detections: {len(detection_data)}, "
                        f"Warnings: {len(warnings)}, "
                        f"Inference: {inference_time:.1f}ms"
                    )
                    
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Failed to send detection data: {e}")

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
                    self.logger.info(f"Object Detection mode changed to: {message}")
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
            self.object_tracker.clear()
            self.object_confidence_history.clear()
            
            if self.debugger:
                self.logger.info(
                    f"Object Detection thread stopped. "
                    f"Processed {self.frame_count} frames, "
                    f"Avg FPS: {self.fps:.1f}"
                )
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Error during shutdown: {e}")
        
        super(threadObjectDetection, self).stop()