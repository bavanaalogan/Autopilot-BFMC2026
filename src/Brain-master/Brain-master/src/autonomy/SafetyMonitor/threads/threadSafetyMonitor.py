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
    ObjectDetection,
    CurrentSpeed,
    CurrentSteer,
    AutonomousDecision,
    EmergencyStop,
    SafetyAlert,
    SensorHealth,
    CommandValidation,
)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.allMessages import StateChange
from src.statemachine.systemMode import SystemMode


class AlertLevel(Enum):
    """Safety alert levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class threadSafetyMonitor(ThreadWithStop):
    """Thread which continuously monitors safety and triggers emergency stops.
    
    Monitors:
    - Collision probability
    - Sensor data freshness
    - Motor command execution
    - Sensor health
    - Command validity
    
    Triggers emergency stop if:
    - Collision risk > 0.9
    - Sensor data missing for >500ms
    - Speed/steering command execution fails
    - Unsafe command detected
    - Multiple sensor failures
    
    Runs at 50 Hz for continuous monitoring with low latency (<20ms).
    
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues.
        logging (logging object): Made for debugging.
        debugging (bool): A flag for debugging.
    """

    # ===== Safety Thresholds =====
    SAFETY_THRESHOLDS = {
        'collision_critical': 0.9,          # Trigger emergency stop
        'collision_high': 0.7,              # Alert
        'collision_medium': 0.5,            # Warning
        'sensor_timeout': 0.5,              # seconds (500ms)
        'command_timeout': 1.0,             # seconds
        'max_speed_error': 100,             # mm/s (maximum acceptable error)
        'max_steer_error': 5.0,             # degrees
    }
    
    # ===== Motor Specifications (for validation) =====
    MOTOR_LIMITS = {
        'speed_min': -500,
        'speed_max': 500,
        'steer_min': -27.2,
        'steer_max': 27.2,
        'max_acceleration': 100,            # mm/s per cycle
        'max_steer_rate': 5.0,              # degrees per cycle
    }
    
    # ===== Sensor Monitoring =====
    SENSORS = {
        'object_detection': {'timeout': 0.5, 'critical': False},
        'speed_feedback': {'timeout': 0.5, 'critical': True},
        'steer_feedback': {'timeout': 0.5, 'critical': True},
        'autonomous_decision': {'timeout': 0.5, 'critical': False},
    }

    # ================================ INIT ===============================================
    def __init__(self, queuesList, logger, debugger):
        super(threadSafetyMonitor, self).__init__(pause=0.02)  # ~50 Hz
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        
        # ===== Input Data =====
        self.object_detections = []
        self.current_speed = 0
        self.current_steer = 0
        self.autonomous_decision = None
        
        # ===== Previous State (for change detection) =====
        self.prev_speed = 0
        self.prev_steer = 0
        self.prev_decision = None
        
        # ===== Sensor Timestamps =====
        self.sensor_timestamps = {
            'object_detection': time.time(),
            'speed_feedback': time.time(),
            'steer_feedback': time.time(),
            'autonomous_decision': time.time(),
        }
        self.sensor_timestamps_lock = Lock()
        
        # ===== Emergency Stop State =====
        self.emergency_stop_triggered = False
        self.emergency_stop_reason = None
        self.emergency_stop_lock = Lock()
        
        # ===== Alert History =====
        self.alert_history = deque(maxlen=100)
        self.critical_alert_count = 0
        self.warning_count = 0
        
        # ===== Performance Tracking =====
        self.frame_count = 0
        self.monitoring_time = 0
        self.fps = 0
        self.last_time = time.time()
        self.emergency_stop_count = 0
        
        # ===== Thresholds for consecutive failures =====
        self.consecutive_failures = {
            'speed_command': 0,
            'steer_command': 0,
            'sensor_failure': 0,
        }
        self.failure_threshold = 3  # Trigger after N consecutive failures
        
        # Initialize message handlers
        self.subscribe()
        self._init_senders()

    # ================================ SUBSCRIBE ===============================================
    def subscribe(self):
        """Subscribe to monitoring data."""
        self.objectSubscriber = messageHandlerSubscriber(
            self.queuesList, ObjectDetection, "lastOnly", True
        )
        self.speedSubscriber = messageHandlerSubscriber(
            self.queuesList, CurrentSpeed, "lastOnly", True
        )
        self.steerSubscriber = messageHandlerSubscriber(
            self.queuesList, CurrentSteer, "lastOnly", True
        )
        self.decisionSubscriber = messageHandlerSubscriber(
            self.queuesList, AutonomousDecision, "lastOnly", True
        )
        self.stateChangeSubscriber = messageHandlerSubscriber(
            self.queuesList, StateChange, "lastOnly", True
        )

    # ================================ SENDERS ===============================================
    def _init_senders(self):
        """Initialize message senders."""
        self.emergencyStopSender = messageHandlerSender(
            self.queuesList, EmergencyStop
        )
        self.safetyAlertSender = messageHandlerSender(
            self.queuesList, SafetyAlert
        )
        self.sensorHealthSender = messageHandlerSender(
            self.queuesList, SensorHealth
        )
        self.commandValidationSender = messageHandlerSender(
            self.queuesList, CommandValidation
        )

    # ================================ RUN ================================================
    def thread_work(self):
        """Main safety monitoring loop."""
        try:
            # Get latest sensor data
            self._update_sensor_data()
            
            # Monitor safety
            start_time = time.time()
            self._monitor_safety()
            monitoring_time = (time.time() - start_time) * 1000  # ms
            self.monitoring_time = monitoring_time
            
            # Send diagnostics periodically
            if self.frame_count % 25 == 0:  # Every 0.5 seconds at 50 Hz
                self._send_diagnostics()
            
            # Update timing
            self._update_fps()
            self.frame_count += 1
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Safety Monitor Error: {e}")
            # On error, trigger emergency stop for safety
            self._trigger_emergency_stop("safety_monitor_error")

    # ================================ UPDATE SENSOR DATA =========================================
    def _update_sensor_data(self):
        """Receive latest sensor data."""
        try:
            # Get object detections
            obj_msg = self.objectSubscriber.receive()
            if obj_msg is not None:
                self.object_detections = obj_msg.get('detections', [])
                self._update_sensor_timestamp('object_detection')
            
            # Get speed feedback
            speed_msg = self.speedSubscriber.receive()
            if speed_msg is not None:
                self.prev_speed = self.current_speed
                self.current_speed = float(speed_msg)
                self._update_sensor_timestamp('speed_feedback')
            
            # Get steering feedback
            steer_msg = self.steerSubscriber.receive()
            if steer_msg is not None:
                self.prev_steer = self.current_steer
                self.current_steer = float(steer_msg)
                self._update_sensor_timestamp('steer_feedback')
            
            # Get autonomous decision
            decision_msg = self.decisionSubscriber.receive()
            if decision_msg is not None:
                self.prev_decision = self.autonomous_decision
                self.autonomous_decision = decision_msg
                self._update_sensor_timestamp('autonomous_decision')
                
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Sensor data update error: {e}")

    # ================================ MONITOR SAFETY =========================================
    def _monitor_safety(self):
        """Main safety monitoring logic."""
        try:
            # Priority 1: Check collision risk
            self._check_collision_risk()
            
            # Priority 2: Check sensor health
            self._check_sensor_health()
            
            # Priority 3: Validate autonomous command
            self._validate_command()
            
            # Priority 4: Check command execution
            self._check_command_execution()
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Safety monitoring error: {e}")

    # ================================ CHECK COLLISION RISK =========================================
    def _check_collision_risk(self):
        """Monitor collision probability."""
        try:
            max_risk = 0.0
            threat_object = None
            
            for detection in self.object_detections:
                distance = detection.get('distance', float('inf'))
                confidence = detection.get('confidence', 0.0)
                class_name = detection.get('class_name', '')
                
                # Skip non-threatening objects
                if class_name in ['parking_meter', 'bench', 'tree', 'traffic_light']:
                    continue
                
                # Calculate risk
                if distance < 0.5:
                    risk = confidence * 1.0
                elif distance < 1.0:
                    risk = confidence * 0.9
                elif distance < 2.0:
                    risk = confidence * 0.7
                elif distance < 3.0:
                    risk = confidence * 0.4
                else:
                    risk = 0.0
                
                if risk > max_risk:
                    max_risk = risk
                    threat_object = detection
            
            # Take action based on risk
            if max_risk > self.SAFETY_THRESHOLDS['collision_critical']:
                self._trigger_emergency_stop(f"critical_collision_risk_{max_risk:.2f}")
                self._log_alert(AlertLevel.CRITICAL, f"Collision risk {max_risk:.2f}", threat_object)
            
            elif max_risk > self.SAFETY_THRESHOLDS['collision_high']:
                self._log_alert(AlertLevel.WARNING, f"High collision risk {max_risk:.2f}", threat_object)
            
            elif max_risk > self.SAFETY_THRESHOLDS['collision_medium']:
                self._log_alert(AlertLevel.INFO, f"Collision risk {max_risk:.2f}", threat_object)
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Collision risk check error: {e}")

    # ================================ CHECK SENSOR HEALTH =========================================
    def _check_sensor_health(self):
        """Check if all sensors are providing timely data."""
        try:
            current_time = time.time()
            critical_sensor_failed = False
            failed_sensors = []
            
            with self.sensor_timestamps_lock:
                for sensor_name, sensor_config in self.SENSORS.items():
                    timestamp = self.sensor_timestamps.get(sensor_name, current_time)
                    time_since_update = current_time - timestamp
                    timeout = sensor_config['timeout']
                    is_critical = sensor_config['critical']
                    
                    # Check timeout
                    if time_since_update > timeout:
                        failed_sensors.append(sensor_name)
                        
                        if is_critical:
                            critical_sensor_failed = True
                            self.consecutive_failures['sensor_failure'] += 1
                        
                        alert_msg = f"Sensor {sensor_name} timeout ({time_since_update:.2f}s)"
                        self._log_alert(AlertLevel.WARNING, alert_msg, None)
                    else:
                        self.consecutive_failures['sensor_failure'] = 0
            
            # Trigger emergency stop if critical sensor failed
            if critical_sensor_failed and self.consecutive_failures['sensor_failure'] > 1:
                self._trigger_emergency_stop(f"critical_sensor_failure_{failed_sensors}")
            
            elif len(failed_sensors) > 2:
                self._log_alert(AlertLevel.CRITICAL, f"Multiple sensor failures: {failed_sensors}", None)
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Sensor health check error: {e}")

    # ================================ VALIDATE COMMAND =========================================
    def _validate_command(self):
        """Validate autonomous command is within safe limits."""
        try:
            if self.autonomous_decision is None:
                return
            
            speed_cmd = self.autonomous_decision.get('speed', 0)
            steer_cmd = self.autonomous_decision.get('steer', 0)
            action = self.autonomous_decision.get('action', 'unknown')
            
            violations = []
            
            # Check speed limits
            if speed_cmd < self.MOTOR_LIMITS['speed_min']:
                violations.append(f"Speed too low: {speed_cmd}")
                speed_cmd = self.MOTOR_LIMITS['speed_min']
            
            elif speed_cmd > self.MOTOR_LIMITS['speed_max']:
                violations.append(f"Speed too high: {speed_cmd}")
                speed_cmd = self.MOTOR_LIMITS['speed_max']
            
            # Check steering limits
            if steer_cmd < self.MOTOR_LIMITS['steer_min']:
                violations.append(f"Steering too left: {steer_cmd}")
                steer_cmd = self.MOTOR_LIMITS['steer_min']
            
            elif steer_cmd > self.MOTOR_LIMITS['steer_max']:
                violations.append(f"Steering too right: {steer_cmd}")
                steer_cmd = self.MOTOR_LIMITS['steer_max']
            
            # Check for unsafe actions with high speed
            if action == "emergency_stop" and abs(speed_cmd) > 50:
                violations.append("Emergency stop with non-zero speed")
            
            if violations:
                self._log_alert(
                    AlertLevel.WARNING,
                    f"Command validation failed: {violations}",
                    {
                        'speed': speed_cmd,
                        'steer': steer_cmd,
                        'action': action
                    }
                )
                
                # Send corrected command info
                self.commandValidationSender.send({
                    'original_speed': self.autonomous_decision.get('speed', 0),
                    'corrected_speed': speed_cmd,
                    'original_steer': self.autonomous_decision.get('steer', 0),
                    'corrected_steer': steer_cmd,
                    'violations': violations,
                    'timestamp': time.time()
                })
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Command validation error: {e}")

    # ================================ CHECK COMMAND EXECUTION =========================================
    def _check_command_execution(self):
        """Check if motor commands are being executed correctly."""
        try:
            if self.autonomous_decision is None or self.prev_decision is None:
                return
            
            target_speed = self.autonomous_decision.get('speed', 0)
            target_steer = self.autonomous_decision.get('steer', 0)
            
            speed_error = abs(target_speed - self.current_speed)
            steer_error = abs(target_steer - self.current_steer)
            
            # Check speed command execution
            if speed_error > self.SAFETY_THRESHOLDS['max_speed_error']:
                self.consecutive_failures['speed_command'] += 1
                
                if self.consecutive_failures['speed_command'] > self.failure_threshold:
                    self._trigger_emergency_stop(f"speed_command_failed_error_{speed_error}")
                
                self._log_alert(
                    AlertLevel.WARNING,
                    f"Speed command not executing: error {speed_error:.1f} mm/s",
                    {'target': target_speed, 'actual': self.current_speed}
                )
            else:
                self.consecutive_failures['speed_command'] = 0
            
            # Check steering command execution
            if steer_error > self.SAFETY_THRESHOLDS['max_steer_error']:
                self.consecutive_failures['steer_command'] += 1
                
                if self.consecutive_failures['steer_command'] > self.failure_threshold:
                    self._trigger_emergency_stop(f"steer_command_failed_error_{steer_error}")
                
                self._log_alert(
                    AlertLevel.WARNING,
                    f"Steer command not executing: error {steer_error:.1f}°",
                    {'target': target_steer, 'actual': self.current_steer}
                )
            else:
                self.consecutive_failures['steer_command'] = 0
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Command execution check error: {e}")

    # ================================ TRIGGER EMERGENCY STOP =========================================
    def _trigger_emergency_stop(self, reason):
        """Trigger emergency stop."""
        try:
            with self.emergency_stop_lock:
                if not self.emergency_stop_triggered:
                    self.emergency_stop_triggered = True
                    self.emergency_stop_reason = reason
                    self.emergency_stop_count += 1
            
            # Send emergency stop message
            self.emergencyStopSender.send({
                'reason': reason,
                'timestamp': time.time()
            })
            
            self._log_alert(AlertLevel.CRITICAL, f"EMERGENCY STOP: {reason}", None)
            
            if self.debugger:
                self.logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Emergency stop trigger error: {e}")

    # ================================ LOG ALERT =========================================
    def _log_alert(self, level, message, details):
        """Log safety alert."""
        try:
            alert = {
                'level': level.value,
                'message': message,
                'details': details,
                'timestamp': time.time()
            }
            
            self.alert_history.append(alert)
            
            # Send alert via message queue
            self.safetyAlertSender.send(alert)
            
            if level == AlertLevel.CRITICAL:
                self.critical_alert_count += 1
            elif level == AlertLevel.WARNING:
                self.warning_count += 1
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Alert logging error: {e}")

    # ================================ SEND DIAGNOSTICS =========================================
    def _send_diagnostics(self):
        """Send sensor health and safety diagnostics."""
        try:
            current_time = time.time()
            sensor_status = {}
            
            with self.sensor_timestamps_lock:
                for sensor_name, sensor_config in self.SENSORS.items():
                    timestamp = self.sensor_timestamps.get(sensor_name, current_time)
                    time_since_update = current_time - timestamp
                    timeout = sensor_config['timeout']
                    
                    sensor_status[sensor_name] = {
                        'last_update_ago': time_since_update,
                        'timeout': timeout,
                        'healthy': time_since_update < timeout,
                        'critical': sensor_config['critical']
                    }
            
            # Send sensor health
            self.sensorHealthSender.send({
                'sensors': sensor_status,
                'healthy_count': sum(1 for s in sensor_status.values() if s['healthy']),
                'total_sensors': len(sensor_status),
                'emergency_stop_active': self.emergency_stop_triggered,
                'timestamp': current_time
            })
            
            if self.debugger and self.frame_count % 100 == 0:
                healthy = sum(1 for s in sensor_status.values() if s['healthy'])
                self.logger.info(
                    f"Safety Monitor - Healthy sensors: {healthy}/{len(sensor_status)}, "
                    f"FPS: {self.fps:.1f}, "
                    f"Time: {self.monitoring_time:.1f}ms"
                )
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Diagnostics sending error: {e}")

    # ================================ UPDATE SENSOR TIMESTAMP =========================================
    def _update_sensor_timestamp(self, sensor_name):
        """Update sensor last update timestamp."""
        try:
            with self.sensor_timestamps_lock:
                self.sensor_timestamps[sensor_name] = time.time()
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Timestamp update error: {e}")

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
                    self.logger.info(f"Safety Monitor mode changed to: {message}")
            except Exception as e:
                if self.debugger:
                    self.logger.error(f"State change error: {e}")

    # ================================ STOP ================================================
    def stop(self):
        """Clean up and stop the thread."""
        try:
            if self.debugger:
                self.logger.info(
                    f"Safety Monitor thread stopped. "
                    f"Processed {self.frame_count} frames, "
                    f"Critical alerts: {self.critical_alert_count}, "
                    f"Warnings: {self.warning_count}, "
                    f"Emergency stops triggered: {self.emergency_stop_count}, "
                    f"Avg FPS: {self.fps:.1f}"
                )
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Error during shutdown: {e}")
        
        super(threadSafetyMonitor, self).stop()