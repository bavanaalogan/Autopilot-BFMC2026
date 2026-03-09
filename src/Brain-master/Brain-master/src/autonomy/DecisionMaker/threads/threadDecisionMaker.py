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

import time
import numpy as np
from collections import deque
from enum import Enum
from threading import Lock

from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (
    LaneDetection,
    ObjectDetection,
    SignDetection,
    Location,
    AutonomousDecision,
    EmergencyStop,
    DecisionMakerStats,
    AutonomyState,
    DecisionLog,
)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.allMessages import StateChange
from src.statemachine.systemMode import SystemMode


class DrivingState(Enum):
    """Autonomous driving states."""
    IDLE = "idle"
    NORMAL_DRIVING = "normal_driving"
    LANE_FOLLOWING = "lane_following"
    OBSTACLE_DETECTED = "obstacle_detected"
    EMERGENCY_STOP = "emergency_stop"
    PARKING = "parking"
    STOPPED = "stopped"
    STOP_SIGN = "stop_sign"
    SPEED_ADJUSTMENT = "speed_adjustment"
    HIGHWAY_MODE = "highway_mode"


class threadDecisionMaker(ThreadWithStop):
    """Thread which makes autonomous driving decisions based on sensor inputs.
    
    Integrates inputs from:
    - Lane detection (lane offset, confidence)
    - Object detection (obstacles, collision risk, distances)
    - Sign detection (stop, speed limits, parking, highway signs)
    - GPS/Location data (position, route information)
    
    Decision logic:
    1. Emergency stop if collision risk is high
    2. Stop at stop signs (3 second stop)
    3. Execute parking maneuvers at parking signs
    4. Adjust speed based on speed limit signs
    5. Increase speed at highway entries
    6. Decrease speed at highway exits
    7. Maintain lanes using lane detection offset
    8. Default to lane following
    
    Outputs:
    - Speed commands (0-500 mm/s, can be negative for reverse)
    - Steering commands (±27.2 degrees)
    - Actions (normal, emergency_stop, park, etc.)
    
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues.
        logging (logging object): Made for debugging.
        debugging (bool): A flag for debugging.
    """

    # ===== Speed Limits (mm/s) =====
    SPEED_LIMITS = {
        'speed_limit_30': 100,      # 30 km/h = ~83 mm/s, round to 100
        'speed_limit_50': 167,      # 50 km/h = ~139 mm/s, round to 167
        'speed_limit_90': 300,      # 90 km/h = ~250 mm/s, round to 300
        'highway_entry': 300,       # Enter highway at moderate speed
        'highway_exit': 167,        # Exit highway at safe speed
        'default': 200,             # Default driving speed
        'min': 50,                  # Minimum creep speed
        'max': 350,                 # Maximum safe speed
    }
    
    # ===== Steering Limits (degrees) =====
    STEERING_LIMITS = {
        'min': -27.2,
        'max': 27.2,
    }
    
    # ===== Collision Detection Thresholds =====
    COLLISION_THRESHOLDS = {
        'critical': 1.0,            # Less than 1 meter
        'high': 2.0,                # Less than 2 meters
        'medium': 3.0,              # Less than 3 meters
        'low': 5.0,                 # Less than 5 meters
    }
    
    # ===== Lane Following Parameters =====
    LANE_FOLLOWING = {
        'max_offset_pixels': 100,   # Max deviation from center
        'offset_to_steer_ratio': 0.2,  # Pixel offset to steering angle ratio
        'smoothing_factor': 0.7,    # Smooth steering transitions
    }
    
    # ===== Decision Making Timing =====
    TIMING = {
        'stop_sign_duration': 3.0,  # Seconds to stop at stop sign
        'deceleration_rate': 50,    # mm/s per decision cycle
        'acceleration_rate': 25,    # mm/s per decision cycle
        'lane_correction_rate': 2,  # Degrees per decision cycle
    }

    # ================================ INIT ===============================================
    def __init__(self, queuesList, logger, debugger):
        super(threadDecisionMaker, self).__init__(pause=0.1)  # ~10 Hz
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        
        # ===== Current Sensor Data =====
        self.lane_detection = None
        self.object_detections = []
        self.sign_detection = None
        self.gps_location = None
        
        # ===== Decision State =====
        self.current_state = DrivingState.IDLE
        self.previous_state = DrivingState.IDLE
        self.current_speed = 0  # mm/s
        self.current_steer = 0  # degrees
        self.target_speed = 0   # mm/s
        
        # ===== Stop Sign Handling =====
        self.stop_sign_detected = False
        self.stop_sign_start_time = None
        self.stop_sign_lock = Lock()
        
        # ===== Parking State =====
        self.parking_active = False
        self.parking_stage = 0  # 0=idle, 1=reverse, 2=steer, 3=forward, 4=align
        self.parking_start_time = None
        
        # ===== Emergency Stop State =====
        self.emergency_stop_active = False
        self.emergency_stop_reason = None
        
        # ===== Speed Profile =====
        self.speed_history = deque(maxlen=10)
        self.steer_history = deque(maxlen=10)
        
        # ===== Collision Avoidance =====
        self.obstacle_warning_count = 0
        self.obstacle_warning_threshold = 2  # Warnings before action
        
        # ===== Performance Tracking =====
        self.decision_count = 0
        self.frame_count = 0
        self.decision_time = 0
        self.fps = 0
        self.last_time = time.time()
        
        # ===== Decision Log =====
        self.decision_history = deque(maxlen=100)
        
        # Initialize message handlers
        self.subscribe()
        self._init_senders()

    # ================================ SUBSCRIBE ===============================================
    def subscribe(self):
        """Subscribe to sensor data and state changes."""
        self.laneSubscriber = messageHandlerSubscriber(
            self.queuesList, LaneDetection, "lastOnly", True
        )
        self.objectSubscriber = messageHandlerSubscriber(
            self.queuesList, ObjectDetection, "lastOnly", True
        )
        self.signSubscriber = messageHandlerSubscriber(
            self.queuesList, SignDetection, "lastOnly", True
        )
        self.gpsSubscriber = messageHandlerSubscriber(
            self.queuesList, Location, "lastOnly", True
        )
        self.stateChangeSubscriber = messageHandlerSubscriber(
            self.queuesList, StateChange, "lastOnly", True
        )

    # ================================ SENDERS ===============================================
    def _init_senders(self):
        """Initialize message senders for autonomous decisions."""
        self.decisionSender = messageHandlerSender(
            self.queuesList, AutonomousDecision
        )
        self.emergencyStopSender = messageHandlerSender(
            self.queuesList, EmergencyStop
        )
        self.statsSender = messageHandlerSender(
            self.queuesList, DecisionMakerStats
        )
        self.stateSender = messageHandlerSender(
            self.queuesList, AutonomyState
        )
        self.logSender = messageHandlerSender(
            self.queuesList, DecisionLog
        )

    # ================================ RUN ================================================
    def thread_work(self):
        """Main decision making loop."""
        try:
            # Get latest sensor data
            self._update_sensor_data()
            
            # Make decision
            start_time = time.time()
            decision = self._make_decision()
            decision_time = (time.time() - start_time) * 1000  # ms
            self.decision_time = decision_time
            
            # Send decision
            if decision:
                self._send_decision(decision)
                self.decision_count += 1
            
            # Send state update
            self._send_state_update()
            
            # Send statistics periodically
            if self.frame_count % 30 == 0:
                self._send_statistics()
            
            # Update timing
            self._update_fps()
            self.frame_count += 1
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Decision Maker Error: {e}")
            # Trigger emergency stop on error
            self._trigger_emergency_stop("decision_maker_error")

    # ================================ UPDATE SENSOR DATA =========================================
    def _update_sensor_data(self):
        """Receive and store latest sensor data."""
        try:
            # Update lane detection
            lane_msg = self.laneSubscriber.receive()
            if lane_msg is not None:
                self.lane_detection = lane_msg
            
            # Update object detections
            obj_msg = self.objectSubscriber.receive()
            if obj_msg is not None:
                self.object_detections = obj_msg.get('detections', [])
            
            # Update sign detection
            sign_msg = self.signSubscriber.receive()
            if sign_msg is not None:
                self.sign_detection = sign_msg
            
            # Update GPS location
            gps_msg = self.gpsSubscriber.receive()
            if gps_msg is not None:
                self.gps_location = gps_msg
                
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Sensor data update error: {e}")

    # ================================ MAKE DECISION =========================================
    def _make_decision(self):
        """Main decision making logic based on sensor inputs and state."""
        try:
            # Priority 1: Emergency Stop
            collision_risk = self._check_collision_risk()
            if collision_risk > 0.7:
                self._trigger_emergency_stop("high_collision_risk")
                return self._create_decision(0, 0, "emergency_stop")
            
            # Priority 2: Stop Sign
            if self.sign_detection and self.sign_detection.get('class_name') == 'stop':
                return self._handle_stop_sign()
            
            # Priority 3: Parking Sign
            if self.sign_detection and self.sign_detection.get('class_name') == 'parking':
                return self._handle_parking()
            
            # Priority 4: Speed Limit Signs
            if self.sign_detection and 'speed_limit' in self.sign_detection.get('class_name', ''):
                self._handle_speed_limit()
            
            # Priority 5: Highway Entry/Exit
            if self.sign_detection and self.sign_detection.get('class_name') == 'highway_entry':
                self._handle_highway_entry()
            elif self.sign_detection and self.sign_detection.get('class_name') == 'highway_exit':
                self._handle_highway_exit()
            
            # Priority 6: Lane Keeping
            if self.sign_detection and self.sign_detection.get('class_name') == 'lane_keep':
                return self._maintain_lane()
            
            # Default: Lane Following
            return self._follow_lane()
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Decision making error: {e}")
            return None

    # ================================ CHECK COLLISION RISK =========================================
    def _check_collision_risk(self):
        """Calculate collision risk based on detected obstacles."""
        try:
            max_risk = 0.0
            
            for detection in self.object_detections:
                distance = detection.get('distance', float('inf'))
                confidence = detection.get('confidence', 0.0)
                class_name = detection.get('class_name', '')
                
                # Skip non-threatening objects
                if class_name in ['parking_meter', 'bench', 'tree']:
                    continue
                
                # Calculate risk score
                if distance < self.COLLISION_THRESHOLDS['critical']:
                    risk = confidence * 1.0
                elif distance < self.COLLISION_THRESHOLDS['high']:
                    risk = confidence * 0.8
                elif distance < self.COLLISION_THRESHOLDS['medium']:
                    risk = confidence * 0.5
                elif distance < self.COLLISION_THRESHOLDS['low']:
                    risk = confidence * 0.2
                else:
                    risk = 0.0
                
                max_risk = max(max_risk, risk)
            
            return max_risk
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Collision risk calculation error: {e}")
            return 0.0

    # ================================ HANDLE STOP SIGN =========================================
    def _handle_stop_sign(self):
        """Handle stop sign detection - decelerate and stop for 3 seconds."""
        try:
            current_time = time.time()
            
            with self.stop_sign_lock:
                # Start stop sign sequence
                if not self.stop_sign_detected:
                    self.stop_sign_detected = True
                    self.stop_sign_start_time = current_time
                    
                    # Log decision
                    self._log_decision("STOP_SIGN_DETECTED", {
                        'position': self.sign_detection.get('position'),
                        'confidence': self.sign_detection.get('confidence')
                    })
                    
                    return self._create_decision(-100, 0, "stop_sign_decelerate")
                
                # Check if stop time is complete
                elapsed_time = current_time - self.stop_sign_start_time
                
                if elapsed_time < 1.0:
                    # Still decelerating
                    return self._create_decision(-100, 0, "stop_sign_stopping")
                
                elif elapsed_time < (1.0 + self.TIMING['stop_sign_duration']):
                    # Stopped
                    return self._create_decision(0, 0, "stop_sign_stopped")
                
                else:
                    # Resume driving
                    self.stop_sign_detected = False
                    self.stop_sign_start_time = None
                    return self._create_decision(self.SPEED_LIMITS['default'], 0, "stop_sign_resume")
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Stop sign handling error: {e}")
            return self._create_decision(0, 0, "stop_sign_error")

    # ================================ HANDLE PARKING =========================================
    def _handle_parking(self):
        """Handle parking sign - execute parking maneuver."""
        try:
            if not self.parking_active:
                # Start parking sequence
                self.parking_active = True
                self.parking_stage = 0
                self.parking_start_time = time.time()
                
                self._log_decision("PARKING_SIGN_DETECTED", {
                    'position': self.sign_detection.get('position'),
                    'confidence': self.sign_detection.get('confidence')
                })
                
                return self._create_decision(-50, 0, "parking_reverse")
            
            # Execute parking stages
            if self.parking_stage == 0:
                # Reverse into parking spot
                self.parking_stage = 1
                return self._create_decision(-100, 0, "parking_reverse")
            
            elif self.parking_stage == 1:
                # Steer to align
                self.parking_stage = 2
                return self._create_decision(-50, 15, "parking_align")
            
            elif self.parking_stage == 2:
                # Fine adjustment
                self.parking_stage = 3
                return self._create_decision(-20, -5, "parking_fine_tune")
            
            else:
                # Parking complete
                self.parking_active = False
                self._log_decision("PARKING_COMPLETE", {})
                return self._create_decision(0, 0, "parking_complete")
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Parking handling error: {e}")
            self.parking_active = False
            return self._create_decision(0, 0, "parking_error")

    # ================================ HANDLE SPEED LIMIT =========================================
    def _handle_speed_limit(self):
        """Handle speed limit signs - adjust target speed."""
        try:
            class_name = self.sign_detection.get('class_name', '')
            
            if 'speed_limit_30' in class_name:
                target_speed = self.SPEED_LIMITS['speed_limit_30']
            elif 'speed_limit_50' in class_name:
                target_speed = self.SPEED_LIMITS['speed_limit_50']
            elif 'speed_limit_90' in class_name:
                target_speed = self.SPEED_LIMITS['speed_limit_90']
            else:
                target_speed = self.SPEED_LIMITS['default']
            
            self.target_speed = target_speed
            
            self._log_decision("SPEED_LIMIT_SIGN", {
                'sign_type': class_name,
                'target_speed': target_speed,
                'current_speed': self.current_speed
            })
            
            # Gradually adjust speed
            if self.current_speed < target_speed:
                new_speed = min(
                    self.current_speed + self.TIMING['acceleration_rate'],
                    target_speed
                )
            else:
                new_speed = max(
                    self.current_speed - self.TIMING['deceleration_rate'],
                    target_speed
                )
            
            self.current_speed = new_speed
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Speed limit handling error: {e}")

    # ================================ HANDLE HIGHWAY ENTRY =========================================
    def _handle_highway_entry(self):
        """Handle highway entry sign - increase speed gradually."""
        try:
            self.target_speed = self.SPEED_LIMITS['highway_entry']
            
            self._log_decision("HIGHWAY_ENTRY", {
                'current_speed': self.current_speed,
                'target_speed': self.target_speed
            })
            
            # Gradually accelerate
            new_speed = min(
                self.current_speed + self.TIMING['acceleration_rate'] * 2,
                self.target_speed
            )
            
            return self._create_decision(new_speed, 0, "highway_entry")
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Highway entry handling error: {e}")

    # ================================ HANDLE HIGHWAY EXIT =========================================
    def _handle_highway_exit(self):
        """Handle highway exit sign - decrease speed gradually."""
        try:
            self.target_speed = self.SPEED_LIMITS['highway_exit']
            
            self._log_decision("HIGHWAY_EXIT", {
                'current_speed': self.current_speed,
                'target_speed': self.target_speed
            })
            
            # Gradually decelerate
            new_speed = max(
                self.current_speed - self.TIMING['deceleration_rate'] * 2,
                self.target_speed
            )
            
            return self._create_decision(new_speed, 0, "highway_exit")
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Highway exit handling error: {e}")

    # ================================ MAINTAIN LANE =========================================
    def _maintain_lane(self):
        """Maintain lane based on lane detection."""
        try:
            if not self.lane_detection:
                return self._create_decision(self.current_speed, 0, "lane_keep_no_lane")
            
            lane_offset = self.lane_detection.get('vehicle_offset', 0)
            lane_confidence = self.lane_detection.get('confidence', 0)
            
            # Calculate steering correction
            steer = -lane_offset * self.LANE_FOLLOWING['offset_to_steer_ratio']
            steer = np.clip(
                steer,
                self.STEERING_LIMITS['min'],
                self.STEERING_LIMITS['max']
            )
            
            # Smooth steering
            steer = self.current_steer * (1 - self.LANE_FOLLOWING['smoothing_factor']) + \
                   steer * self.LANE_FOLLOWING['smoothing_factor']
            
            self.current_steer = steer
            
            self._log_decision("LANE_KEEP", {
                'lane_offset': lane_offset,
                'steering': steer,
                'confidence': lane_confidence
            })
            
            return self._create_decision(self.current_speed, steer, "lane_keep")
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Lane keep error: {e}")
            return self._create_decision(self.current_speed, 0, "lane_keep_error")

    # ================================ FOLLOW LANE =========================================
    def _follow_lane(self):
        """Default behavior - follow lane."""
        try:
            if not self.lane_detection:
                # No lane detected - slow down
                speed = self.SPEED_LIMITS['min']
                steer = 0
                action = "no_lane_detected"
            else:
                lane_offset = self.lane_detection.get('vehicle_offset', 0)
                lane_confidence = self.lane_detection.get('confidence', 0)
                
                # Calculate steering to stay in lane
                steer = -lane_offset * self.LANE_FOLLOWING['offset_to_steer_ratio']
                steer = np.clip(
                    steer,
                    self.STEERING_LIMITS['min'],
                    self.STEERING_LIMITS['max']
                )
                
                # Smooth steering
                steer = self.current_steer * (1 - self.LANE_FOLLOWING['smoothing_factor']) + \
                       steer * self.LANE_FOLLOWING['smoothing_factor']
                
                self.current_steer = steer
                
                # Use current speed or default
                speed = max(self.current_speed, self.SPEED_LIMITS['default'])
                speed = min(speed, self.SPEED_LIMITS['max'])
                
                action = "lane_following"
            
            self._log_decision("LANE_FOLLOWING", {
                'speed': speed,
                'steering': steer,
                'lane_detected': self.lane_detection is not None
            })
            
            return self._create_decision(speed, steer, action)
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Lane following error: {e}")
            return self._create_decision(0, 0, "lane_following_error")

    # ================================ TRIGGER EMERGENCY STOP =========================================
    def _trigger_emergency_stop(self, reason):
        """Trigger emergency stop due to critical condition."""
        try:
            self.emergency_stop_active = True
            self.emergency_stop_reason = reason
            self.current_state = DrivingState.EMERGENCY_STOP
            
            self._log_decision("EMERGENCY_STOP", {
                'reason': reason
            })
            
            # Send emergency stop message
            self.emergencyStopSender.send({
                'reason': reason,
                'timestamp': time.time()
            })
            
            if self.debugger:
                self.logger.warning(f"EMERGENCY STOP: {reason}")
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Emergency stop error: {e}")

    # ================================ CREATE DECISION =========================================
    def _create_decision(self, speed, steer, action):
        """Create a decision message."""
        try:
            # Clamp values
            speed = np.clip(speed, -self.SPEED_LIMITS['max'], self.SPEED_LIMITS['max'])
            steer = np.clip(steer, self.STEERING_LIMITS['min'], self.STEERING_LIMITS['max'])
            
            # Update state
            self.current_speed = speed
            self.current_steer = steer
            
            # Store in history
            self.speed_history.append(speed)
            self.steer_history.append(steer)
            
            # Create decision
            decision = {
                'speed': float(speed),
                'steer': float(steer),
                'action': action,
                'state': self.current_state.value,
                'sensor_data': {
                    'lane_detected': self.lane_detection is not None,
                    'obstacles_detected': len(self.object_detections) > 0,
                    'sign_detected': self.sign_detection is not None,
                },
                'timestamp': time.time()
            }
            
            return decision
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Decision creation error: {e}")
            return None

    # ================================ SEND DECISION =========================================
    def _send_decision(self, decision):
        """Send autonomous decision via message queue."""
        try:
            self.decisionSender.send(decision)
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Failed to send decision: {e}")

    # ================================ SEND STATE UPDATE =========================================
    def _send_state_update(self):
        """Send current autonomy state."""
        try:
            self.stateSender.send(self.current_state.value)
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Failed to send state: {e}")

    # ================================ SEND STATISTICS =========================================
    def _send_statistics(self):
        """Send decision making statistics."""
        try:
            stats = {
                'decision_count': self.decision_count,
                'frame_count': self.frame_count,
                'decision_time_ms': self.decision_time,
                'fps': self.fps,
                'current_speed': self.current_speed,
                'current_steer': self.current_steer,
                'target_speed': self.target_speed,
                'current_state': self.current_state.value,
                'emergency_stop_active': self.emergency_stop_active,
                'emergency_stop_reason': self.emergency_stop_reason,
                'parking_active': self.parking_active,
                'stop_sign_active': self.stop_sign_detected,
                'lane_detected': self.lane_detection is not None,
                'objects_detected': len(self.object_detections),
                'sign_detected': self.sign_detection is not None,
                'timestamp': time.time()
            }
            
            self.statsSender.send(stats)
            
            if self.debugger:
                self.logger.info(
                    f"Decision Maker - Speed: {self.current_speed:.1f} mm/s, "
                    f"Steer: {self.current_steer:.1f}°, "
                    f"State: {self.current_state.value}, "
                    f"Time: {self.decision_time:.1f}ms"
                )
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Failed to send statistics: {e}")

    # ================================ LOG DECISION =========================================
    def _log_decision(self, decision_type, details):
        """Log decision for debugging and analysis."""
        try:
            log_entry = {
                'decision_type': decision_type,
                'details': details,
                'state': self.current_state.value,
                'speed': self.current_speed,
                'steer': self.current_steer,
                'timestamp': time.time()
            }
            
            self.decision_history.append(log_entry)
            
            # Send log periodically
            if len(self.decision_history) % 10 == 0:
                self.logSender.send(log_entry)
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Decision logging error: {e}")

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
                modeDict = SystemMode[message].value["autonomy"]["thread"]
                
                if self.debugger:
                    self.logger.info(f"Decision Maker mode changed to: {message}")
            except Exception as e:
                if self.debugger:
                    self.logger.error(f"State change error: {e}")

    # ================================ STOP ================================================
    def stop(self):
        """Clean up and stop the thread."""
        try:
            if self.debugger:
                self.logger.info(
                    f"Decision Maker thread stopped. "
                    f"Made {self.decision_count} decisions, "
                    f"Avg FPS: {self.fps:.1f}"
                )
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Error during shutdown: {e}")
        
        super(threadDecisionMaker, self).stop()