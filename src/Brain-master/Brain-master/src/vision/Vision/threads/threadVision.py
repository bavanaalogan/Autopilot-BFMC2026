import cv2
import numpy as np
import base64
import time
from collections import deque

from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (
    mainCamera,
    LaneDetection,
    LanePositionError,
    LaneConfidence,
    ProcessedFrame,
)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.allMessages import StateChange
from src.statemachine.systemMode import SystemMode


class threadLaneDetection(ThreadWithStop):
    """Thread which will handle lane detection using computer vision.
    
    Detects road lanes in real-time using:
    - Canny edge detection
    - Hough transform for line detection
    - Polyline fitting for smooth lane curves
    
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool): A flag for debugging.
    """

    # ================================ INIT ===============================================
    def __init__(self, queuesList, logger, debugger):
        super(threadLaneDetection, self).__init__(pause=0.033)  # ~30 FPS
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        
        # ===== Lane Detection Parameters =====
        self.frame_width = 640
        self.frame_height = 480
        
        # ROI (Region of Interest) for lane detection
        self.roi_top = int(self.frame_height * 0.5)  # Start from middle of frame
        self.roi_bottom = self.frame_height
        self.roi_left = 0
        self.roi_right = self.frame_width
        
        # Canny edge detection parameters
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        
        # Hough line transform parameters
        self.hough_rho = 1
        self.hough_theta = np.pi / 180
        self.hough_threshold = 50
        self.hough_min_length = 30
        self.hough_max_gap = 10
        
        # Lane filtering parameters
        self.lane_angle_min = 20  # degrees
        self.lane_angle_max = 160  # degrees
        
        # Smoothing buffer
        self.frame_buffer_size = 5
        self.lane_history = deque(maxlen=self.frame_buffer_size)
        
        # Initialize message handlers
        self.subscribe()
        self._init_senders()
        
        # Performance tracking
        self.frame_count = 0
        self.processing_time = 0

    # ================================ SUBSCRIBE ===============================================
    def subscribe(self):
        """Subscribe to camera frames and state changes."""
        self.cameraSubriber = messageHandlerSubscriber(
            self.queuesList, mainCamera, "lastOnly", True
        )
        self.stateChangeSubscriber = messageHandlerSubscriber(
            self.queuesList, StateChange, "lastOnly", True
        )

    # ================================ SENDERS ===============================================
    def _init_senders(self):
        """Initialize message senders for lane detection outputs."""
        self.laneDetectionSender = messageHandlerSender(
            self.queuesList, LaneDetection
        )
        self.lanePositionErrorSender = messageHandlerSender(
            self.queuesList, LanePositionError
        )
        self.laneConfidenceSender = messageHandlerSender(
            self.queuesList, LaneConfidence
        )
        self.processedFrameSender = messageHandlerSender(
            self.queuesList, ProcessedFrame
        )

    # ================================ RUN ================================================
    def thread_work(self):
        """Main thread work - capture frame and perform lane detection."""
        try:
            # Receive camera frame
            frame_data = self.cameraSub briber.receive()
            if frame_data is None:
                return
            
            # Decode frame from base64
            frame = self._decode_frame(frame_data)
            if frame is None:
                return
            
            # Perform lane detection
            start_time = time.time()
            lane_data = self._detect_lanes(frame)
            self.processing_time = (time.time() - start_time) * 1000  # ms
            
            # Send detection results
            if lane_data is not None:
                self._send_lane_data(lane_data)
                
            self.frame_count += 1
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Lane Detection Error: {e}")

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

    # ================================ LANE DETECTION =========================================
    def _detect_lanes(self, frame):
        """Perform lane detection on frame using computer vision."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Extract ROI (Region of Interest)
            roi = gray[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right]
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(roi, (5, 5), 1.0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
            
            # Apply morphological operations to connect nearby edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(
                edges,
                self.hough_rho,
                self.hough_theta,
                self.hough_threshold,
                minLineLength=self.hough_min_length,
                maxLineGap=self.hough_max_gap
            )
            
            if lines is None or len(lines) == 0:
                return None
            
            # Filter lines by angle (eliminate near-horizontal lines)
            lane_lines = self._filter_lanes(lines)
            
            if len(lane_lines) == 0:
                return None
            
            # Separate left and right lanes
            left_lanes, right_lanes = self._separate_left_right(lane_lines, roi.shape[1])
            
            # Fit polynomial to lane lines
            left_lane = self._fit_lane(left_lanes, roi.shape[1], roi.shape[0])
            right_lane = self._fit_lane(right_lanes, roi.shape[1], roi.shape[0])
            
            # Calculate center position and offset
            lane_data = self._calculate_lane_metrics(
                left_lane, right_lane, roi.shape[1], frame
            )
            
            # Smooth lane data using history buffer
            self.lane_history.append(lane_data)
            smoothed_data = self._smooth_lane_data()
            
            return smoothed_data
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Lane detection processing error: {e}")
            return None

    # ================================ FILTER LANES =========================================
    def _filter_lanes(self, lines):
        """Filter lines by angle to remove near-horizontal lines."""
        filtered_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.degrees(angle_rad)
            
            # Convert to 0-180 range
            if angle_deg < 0:
                angle_deg += 180
            
            # Filter by angle (avoid near-horizontal lines)
            if self.lane_angle_min <= angle_deg <= self.lane_angle_max:
                filtered_lines.append(line[0])
        
        return filtered_lines

    # ================================ SEPARATE LEFT/RIGHT =========================================
    def _separate_left_right(self, lines, width):
        """Separate detected lines into left and right lanes."""
        left_lanes = []
        right_lanes = []
        center_x = width / 2
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Calculate slope
            if (x2 - x1) == 0:
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            
            # Left lane: negative slope, on left side
            if slope < -0.3:
                if (x1 + x2) / 2 < center_x:
                    left_lanes.append(line)
            
            # Right lane: positive slope, on right side
            elif slope > 0.3:
                if (x1 + x2) / 2 > center_x:
                    right_lanes.append(line)
        
        return left_lanes, right_lanes

    # ================================ FIT LANE =========================================
    def _fit_lane(self, lanes, width, height):
        """Fit polynomial curve to lane lines."""
        if len(lanes) == 0:
            return None
        
        try:
            # Collect all points
            points_x = []
            points_y = []
            
            for x1, y1, x2, y2 in lanes:
                points_x.extend([x1, x2])
                points_y.extend([y1, y2])
            
            if len(points_x) < 4:
                return None
            
            # Fit 2nd degree polynomial
            coeffs = np.polyfit(points_y, points_x, 2)
            
            return {
                'coeffs': coeffs,
                'points_x': points_x,
                'points_y': points_y
            }
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Lane fitting error: {e}")
            return None

    # ================================ CALCULATE LANE METRICS =========================================
    def _calculate_lane_metrics(self, left_lane, right_lane, width, frame):
        """Calculate lane center position and vehicle offset."""
        lane_data = {
            'left_lane': left_lane,
            'right_lane': right_lane,
            'lane_center': None,
            'vehicle_offset': None,
            'lane_width': None,
            'confidence': 0.0,
            'timestamp': time.time()
        }
        
        # Calculate lane center
        if left_lane is not None and right_lane is not None:
            # Get x-coordinates at bottom of ROI
            y_bottom = self.frame_height - self.roi_top - 1
            
            left_x = np.polyval(left_lane['coeffs'], y_bottom)
            right_x = np.polyval(right_lane['coeffs'], y_bottom)
            
            lane_center = (left_x + right_x) / 2
            vehicle_center = width / 2
            
            lane_data['lane_center'] = lane_center
            lane_data['vehicle_offset'] = vehicle_center - lane_center
            lane_data['lane_width'] = abs(right_x - left_x)
            lane_data['confidence'] = 0.9  # Both lanes detected
            
        elif left_lane is not None:
            lane_data['left_lane'] = left_lane
            lane_data['confidence'] = 0.5
            
        elif right_lane is not None:
            lane_data['right_lane'] = right_lane
            lane_data['confidence'] = 0.5
        
        return lane_data

    # ================================ SMOOTH LANE DATA =========================================
    def _smooth_lane_data(self):
        """Smooth lane data using history buffer."""
        if len(self.lane_history) == 0:
            return None
        
        smoothed = self.lane_history[-1].copy()
        
        # Average offset over history
        if len(self.lane_history) > 1:
            offsets = [d['vehicle_offset'] for d in self.lane_history if d['vehicle_offset'] is not None]
            if offsets:
                smoothed['vehicle_offset'] = np.mean(offsets)
        
        return smoothed

    # ================================ SEND LANE DATA =========================================
    def _send_lane_data(self, lane_data):
        """Send lane detection results via message queue."""
        try:
            # Send main lane detection data
            self.laneDetectionSender.send(lane_data)
            
            # Send vehicle position error (offset from lane center)
            if lane_data['vehicle_offset'] is not None:
                self.lanePositionErrorSender.send(lane_data['vehicle_offset'])
            
            # Send confidence level
            self.laneConfidenceSender.send(lane_data['confidence'])
            
            if self.debugger and self.frame_count % 30 == 0:
                self.logger.info(
                    f"Lane Detection - Offset: {lane_data['vehicle_offset']:.1f}px, "
                    f"Confidence: {lane_data['confidence']:.2f}, "
                    f"Processing: {self.processing_time:.1f}ms"
                )
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Failed to send lane data: {e}")

    # ================================ STATE CHANGE HANDLER ========================================
    def state_change_handler(self):
        """Handle state changes from the state machine."""
        message = self.stateChangeSubscriber.receive()
        if message is not None:
            try:
                modeDict = SystemMode[message].value["vision"]["thread"]
                
                if self.debugger:
                    self.logger.info(f"Lane Detection mode changed to: {message}")
            except Exception as e:
                if self.debugger:
                    self.logger.error(f"State change error: {e}")

    # ================================ STOP ================================================
    def stop(self):
        """Clean up and stop the thread."""
        if self.debugger:
            self.logger.info(f"Lane Detection thread stopped. Processed {self.frame_count} frames")
        super(threadLaneDetection, self).stop()