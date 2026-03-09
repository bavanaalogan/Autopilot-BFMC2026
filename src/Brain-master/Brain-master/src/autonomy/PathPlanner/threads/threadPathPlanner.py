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
from threading import Lock

from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (
    LaneDetection,
    ObjectDetection,
    SignDetection,
    PlannedPath,
    Waypoint,
)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.allMessages import StateChange
from src.statemachine.systemMode import SystemMode


class threadPathPlanner(ThreadWithStop):
    """Thread which plans optimal paths based on sensor inputs.
    
    Path planning capabilities:
    - Lane following trajectory
    - Obstacle avoidance
    - Parking maneuver planning
    - Speed profile generation
    - Look-ahead planning (2-3 seconds)
    - Waypoint generation
    
    Updates at 5 Hz for strategic planning decisions.
    
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues.
        logging (logging object): Made for debugging.
        debugging (bool): A flag for debugging.
    """

    # ===== Planning Parameters =====
    PLANNING_PARAMS = {
        'lookahead_time': 3.0,              # seconds (look ahead 3 seconds)
        'lookahead_distance': 300,          # mm (approximately 0.3m)
        'waypoint_spacing': 100,            # mm between waypoints
        'lane_centering_gain': 0.5,         # strength of lane centering
        'obstacle_margin': 0.5,             # meters safety margin around obstacles
        'planning_horizon': 2.0,            # seconds into future to plan
    }
    
    # ===== Speed Profile Parameters =====
    SPEED_PROFILE = {
        'normal_driving': 200,              # mm/s
        'lane_following': 150,              # mm/s
        'obstacle_approach': 100,           # mm/s
        'parking': 50,                      # mm/s
        'stop': 0,                          # mm/s
        'acceleration': 50,                 # mm/s per planning cycle
        'deceleration': 75,                 # mm/s per planning cycle
    }
    
    # ===== Parking Parameters =====
    PARKING = {
        'maneuver_duration': 8.0,           # seconds
        'reverse_distance': 1000,           # mm
        'steer_angle': 27.0,                # degrees
    }

    # ================================ INIT ===============================================
    def __init__(self, queuesList, logger, debugger):
        super(threadPathPlanner, self).__init__(pause=0.2)  # ~5 Hz
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        
        # ===== Input Data =====
        self.lane_detection = None
        self.object_detections = []
        self.sign_detection = None
        
        # ===== Planned Path =====
        self.waypoints = []
        self.speed_profile = []
        self.steering_profile = []
        self.current_waypoint_idx = 0
        
        # ===== Parking State =====
        self.parking_enabled = False
        self.parking_stage = 0
        self.parking_start_time = None
        
        # ===== Path History =====
        self.path_history = deque(maxlen=10)
        
        # ===== Performance Tracking =====
        self.frame_count = 0
        self.planning_time = 0
        self.fps = 0
        self.last_time = time.time()
        self.plan_count = 0
        
        # Initialize message handlers
        self.subscribe()
        self._init_senders()

    # ================================ SUBSCRIBE ===============================================
    def subscribe(self):
        """Subscribe to sensor data."""
        self.laneSubscriber = messageHandlerSubscriber(
            self.queuesList, LaneDetection, "lastOnly", True
        )
        self.objectSubscriber = messageHandlerSubscriber(
            self.queuesList, ObjectDetection, "lastOnly", True
        )
        self.signSubscriber = messageHandlerSubscriber(
            self.queuesList, SignDetection, "lastOnly", True
        )
        self.stateChangeSubscriber = messageHandlerSubscriber(
            self.queuesList, StateChange, "lastOnly", True
        )

    # ================================ SENDERS ===============================================
    def _init_senders(self):
        """Initialize message senders."""
        self.pathSender = messageHandlerSender(
            self.queuesList, PlannedPath
        )
        self.waypointSender = messageHandlerSender(
            self.queuesList, Waypoint
        )

    # ================================ RUN ================================================
    def thread_work(self):
        """Main path planning loop."""
        try:
            # Get latest sensor data
            self._update_sensor_data()
            
            # Plan path
            start_time = time.time()
            self._plan_path()
            planning_time = (time.time() - start_time) * 1000  # ms
            self.planning_time = planning_time
            
            # Send planned path
            self._send_planned_path()
            self.plan_count += 1
            
            # Send diagnostics
            if self.frame_count % 5 == 0:
                self._send_diagnostics()
            
            # Update timing
            self._update_fps()
            self.frame_count += 1
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Path Planner Error: {e}")

    # ================================ UPDATE SENSOR DATA =========================================
    def _update_sensor_data(self):
        """Receive latest sensor data."""
        try:
            # Get lane detection
            lane_msg = self.laneSubscriber.receive()
            if lane_msg is not None:
                self.lane_detection = lane_msg
            
            # Get object detections
            obj_msg = self.objectSubscriber.receive()
            if obj_msg is not None:
                self.object_detections = obj_msg.get('detections', [])
            
            # Get sign detection
            sign_msg = self.signSubscriber.receive()
            if sign_msg is not None:
                self.sign_detection = sign_msg
                
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Sensor data update error: {e}")

    # ================================ PLAN PATH =========================================
    def _plan_path(self):
        """Main path planning logic."""
        try:
            # Clear previous plan
            self.waypoints = []
            self.speed_profile = []
            self.steering_profile = []
            
            # Check for parking sign
            if self.sign_detection and self.sign_detection.get('class_name') == 'parking':
                self._plan_parking_maneuver()
                return
            
            # Default: Plan lane following path
            self._plan_lane_following_path()
            
            # Adjust for obstacles
            self._adjust_for_obstacles()
            
            # Generate speed profile
            self._generate_speed_profile()
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Path planning error: {e}")

    # ================================ PLAN LANE FOLLOWING PATH =========================================
    def _plan_lane_following_path(self):
        """Plan trajectory following detected lanes."""
        try:
            if self.lane_detection is None:
                # No lane detected - move forward cautiously
                self.waypoints = [
                    {'x': 0, 'y': 0, 'heading': 0},
                    {'x': 300, 'y': 0, 'heading': 0},
                    {'x': 600, 'y': 0, 'heading': 0},
                ]
                return
            
            # Get lane information
            lane_offset = self.lane_detection.get('vehicle_offset', 0)
            lane_width = self.lane_detection.get('lane_width', 200)
            lane_confidence = self.lane_detection.get('confidence', 0)
            
            # Generate waypoints along lane
            num_waypoints = 6
            for i in range(num_waypoints):
                distance = (i + 1) * (self.PLANNING_PARAMS['lookahead_distance'] / num_waypoints)
                
                # Calculate lateral offset to center lane
                # Gradual centering
                lateral_offset = lane_offset * (1 - i / num_waypoints) * self.PLANNING_PARAMS['lane_centering_gain']
                
                # Limit lateral movement
                lateral_offset = np.clip(lateral_offset, -50, 50)
                
                waypoint = {
                    'x': distance,
                    'y': lateral_offset,
                    'heading': np.arctan2(lateral_offset, distance) * 180 / np.pi,
                    'confidence': lane_confidence,
                }
                
                self.waypoints.append(waypoint)
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Lane following planning error: {e}")

    # ================================ ADJUST FOR OBSTACLES =========================================
    def _adjust_for_obstacles(self):
        """Adjust planned path to avoid obstacles."""
        try:
            if not self.waypoints or not self.object_detections:
                return
            
            for detection in self.object_detections:
                distance = detection.get('distance', float('inf'))
                position = detection.get('position', {})
                class_name = detection.get('class_name', '')
                
                # Skip non-blocking obstacles
                if distance > 2.0 or class_name in ['parking_meter', 'bench']:
                    continue
                
                # Calculate obstacle position relative to vehicle
                obstacle_x = distance * 1000  # Convert to mm
                obstacle_y_offset = position.get('centroid_x', 0) - 320  # Assuming 640 width
                
                # Adjust waypoints to avoid obstacle
                for i, waypoint in enumerate(self.waypoints):
                    if waypoint['x'] > obstacle_x - 500:  # 500mm buffer before obstacle
                        # Shift waypoint laterally
                        if obstacle_y_offset < 0:
                            # Obstacle on left, shift right
                            waypoint['y'] += 100
                        else:
                            # Obstacle on right, shift left
                            waypoint['y'] -= 100
                        
                        # Update heading
                        if i > 0:
                            prev_wp = self.waypoints[i-1]
                            dy = waypoint['y'] - prev_wp['y']
                            dx = waypoint['x'] - prev_wp['x']
                            waypoint['heading'] = np.arctan2(dy, dx) * 180 / np.pi
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Obstacle avoidance error: {e}")

    # ================================ PLAN PARKING MANEUVER =========================================
    def _plan_parking_maneuver(self):
        """Plan automated parking maneuver."""
        try:
            self.parking_enabled = True
            self.parking_stage = 0
            self.parking_start_time = time.time()
            
            # Stage 1: Reverse into spot
            self.waypoints = [
                {'x': 0, 'y': 0, 'heading': 0, 'speed': 50},
                {'x': -500, 'y': 0, 'heading': 0, 'speed': 50},
                {'x': -1000, 'y': 0, 'heading': 0, 'speed': 30},
            ]
            
            if self.debugger:
                self.logger.info("Parking maneuver planned")
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Parking planning error: {e}")

    # ================================ GENERATE SPEED PROFILE =========================================
    def _generate_speed_profile(self):
        """Generate speed profile for planned path."""
        try:
            self.speed_profile = []
            
            for i, waypoint in enumerate(self.waypoints):
                # Check for obstacles near this waypoint
                safe_speed = self.SPEED_PROFILE['normal_driving']
                
                for detection in self.object_detections:
                    distance = detection.get('distance', float('inf'))
                    
                    if distance < 1.0:
                        safe_speed = self.SPEED_PROFILE['obstacle_approach']
                    elif distance < 2.0:
                        safe_speed = min(safe_speed, self.SPEED_PROFILE['lane_following'])
                
                # Parking speed if in parking mode
                if self.parking_enabled:
                    safe_speed = self.SPEED_PROFILE['parking']
                
                self.speed_profile.append(safe_speed)
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Speed profile generation error: {e}")

    # ================================ SEND PLANNED PATH =========================================
    def _send_planned_path(self):
        """Send planned path via message queue."""
        try:
            if not self.waypoints:
                return
            
            # Send main path
            path_data = {
                'waypoints': self.waypoints,
                'speed_profile': self.speed_profile,
                'total_distance': sum([wp.get('x', 0) for wp in self.waypoints]),
                'parking_enabled': self.parking_enabled,
                'timestamp': time.time()
            }
            
            self.pathSender.send(path_data)
            
            # Send first waypoint
            if self.waypoints:
                self.waypointSender.send({
                    'index': 0,
                    'waypoint': self.waypoints[0],
                    'target_speed': self.speed_profile[0] if self.speed_profile else 0,
                    'timestamp': time.time()
                })
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Failed to send path: {e}")

    # ================================ SEND DIAGNOSTICS =========================================
    def _send_diagnostics(self):
        """Send path planning diagnostics."""
        try:
            if self.debugger and self.frame_count % 10 == 0:
                self.logger.info(
                    f"Path Planner - Waypoints: {len(self.waypoints)}, "
                    f"Speed profile points: {len(self.speed_profile)}, "
                    f"Obstacles avoided: {len(self.object_detections)}, "
                    f"Planning time: {self.planning_time:.1f}ms, "
                    f"FPS: {self.fps:.1f}"
                )
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Diagnostics error: {e}")

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
                    self.logger.info(f"Path Planner mode changed to: {message}")
            except Exception as e:
                if self.debugger:
                    self.logger.error(f"State change error: {e}")

    # ================================ STOP ================================================
    def stop(self):
        """Clean up and stop the thread."""
        try:
            if self.debugger:
                self.logger.info(
                    f"Path Planner thread stopped. "
                    f"Generated {self.plan_count} plans, "
                    f"Avg FPS: {self.fps:.1f}"
                )
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Error during shutdown: {e}")
        
        super(threadPathPlanner, self).stop()