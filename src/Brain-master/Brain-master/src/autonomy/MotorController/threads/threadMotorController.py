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
    AutonomousDecision,
    CurrentSpeed,
    CurrentSteer,
    SpeedMotor,
    SteerMotor,
    MotorCommandStats,
    PIDFeedback,
    MotorCommand,
)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.allMessages import StateChange
from src.statemachine.systemMode import SystemMode


class PIDController:
    """Simple PID controller for smooth control."""
    
    def __init__(self, kp, ki, kd, setpoint=0, output_limits=None):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            setpoint: Target value
            output_limits: Tuple of (min, max) output limits
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits or (-float('inf'), float('inf'))
        
        # State variables
        self.integral = 0
        self.last_error = 0
        self.last_time = time.time()
        self.lock = Lock()
    
    def update(self, current_value):
        """Calculate PID output."""
        with self.lock:
            current_time = time.time()
            dt = current_time - self.last_time
            
            if dt <= 0:
                return 0
            
            # Calculate error
            error = self.setpoint - current_value
            
            # Proportional term
            p_term = self.kp * error
            
            # Integral term (with anti-windup)
            self.integral += error * dt
            max_integral = self.output_limits[1] / (self.ki + 1e-6) if self.ki != 0 else float('inf')
            self.integral = np.clip(self.integral, -max_integral, max_integral)
            i_term = self.ki * self.integral
            
            # Derivative term
            if dt > 0:
                d_term = self.kd * (error - self.last_error) / dt
            else:
                d_term = 0
            
            # Sum and limit output
            output = p_term + i_term + d_term
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
            
            # Store for next iteration
            self.last_error = error
            self.last_time = current_time
            
            return output
    
    def set_setpoint(self, setpoint):
        """Update setpoint."""
        with self.lock:
            self.setpoint = setpoint
    
    def reset(self):
        """Reset PID state."""
        with self.lock:
            self.integral = 0
            self.last_error = 0
            self.last_time = time.time()


class threadMotorController(ThreadWithStop):
    """Thread which converts autonomous decisions to motor commands with PID control.
    
    Responsibilities:
    - Receive autonomous driving decisions
    - Get current speed and steering feedback
    - Use PID controllers for smooth control
    - Convert to motor PWM commands
    - Send commands to serial handler (NUCLEO board)
    - Monitor command execution
    - Handle emergency stops
    
    Motor Specifications:
    - Speed: 0-500 mm/s (can be negative for reverse)
    - Steering: ±27.2 degrees
    - Both controlled via PWM over serial
    
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues.
        logging (logging object): Made for debugging.
        debugging (bool): A flag for debugging.
    """

    # ===== Motor Specifications =====
    MOTOR_SPECS = {
        'speed': {
            'min': -500,            # mm/s (reverse)
            'max': 500,             # mm/s (forward)
            'neutral': 0,
            'ramp_rate': 50,        # mm/s per cycle
        },
        'steering': {
            'min': -27.2,           # degrees (left)
            'max': 27.2,            # degrees (right)
            'neutral': 0,
            'ramp_rate': 2.0,       # degrees per cycle
        }
    }
    
    # ===== PID Tuning Parameters =====
    # These should be tuned based on your robot's dynamics
    PID_PARAMS = {
        'speed': {
            'kp': 0.8,              # Proportional gain
            'ki': 0.1,              # Integral gain
            'kd': 0.05,             # Derivative gain
            'output_limits': (-500, 500),
        },
        'steering': {
            'kp': 1.0,
            'ki': 0.05,
            'kd': 0.1,
            'output_limits': (-27.2, 27.2),
        }
    }

    # ================================ INIT ===============================================
    def __init__(self, queuesList, logger, debugger):
        super(threadMotorController, self).__init__(pause=0.05)  # ~20 Hz
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        
        # ===== Input Data =====
        self.autonomous_decision = None
        self.current_speed_feedback = 0
        self.current_steer_feedback = 0
        
        # ===== Motor Commands =====
        self.last_speed_command = 0
        self.last_steer_command = 0
        self.speed_command = 0
        self.steer_command = 0
        
        # ===== PID Controllers =====
        self.speed_pid = PIDController(
            kp=self.PID_PARAMS['speed']['kp'],
            ki=self.PID_PARAMS['speed']['ki'],
            kd=self.PID_PARAMS['speed']['kd'],
            setpoint=0,
            output_limits=self.PID_PARAMS['speed']['output_limits']
        )
        
        self.steer_pid = PIDController(
            kp=self.PID_PARAMS['steering']['kp'],
            ki=self.PID_PARAMS['steering']['ki'],
            kd=self.PID_PARAMS['steering']['kd'],
            setpoint=0,
            output_limits=self.PID_PARAMS['steering']['output_limits']
        )
        
        # ===== Control Smoothing =====
        self.speed_smoothing_factor = 0.7
        self.steer_smoothing_factor = 0.8
        
        # ===== Emergency Stop State =====
        self.emergency_stop_active = False
        self.emergency_stop_lock = Lock()
        
        # ===== Command History =====
        self.speed_history = deque(maxlen=20)
        self.steer_history = deque(maxlen=20)
        
        # ===== Performance Tracking =====
        self.frame_count = 0
        self.command_count = 0
        self.processing_time = 0
        self.fps = 0
        self.last_time = time.time()
        
        # ===== Error Tracking =====
        self.speed_error_history = deque(maxlen=20)
        self.steer_error_history = deque(maxlen=20)
        
        # Initialize message handlers
        self.subscribe()
        self._init_senders()

    # ================================ SUBSCRIBE ===============================================
    def subscribe(self):
        """Subscribe to autonomous decisions and motor feedback."""
        self.decisionSubscriber = messageHandlerSubscriber(
            self.queuesList, AutonomousDecision, "lastOnly", True
        )
        self.speedFeedbackSubscriber = messageHandlerSubscriber(
            self.queuesList, CurrentSpeed, "lastOnly", True
        )
        self.steerFeedbackSubscriber = messageHandlerSubscriber(
            self.queuesList, CurrentSteer, "lastOnly", True
        )
        self.stateChangeSubscriber = messageHandlerSubscriber(
            self.queuesList, StateChange, "lastOnly", True
        )

    # ================================ SENDERS ===============================================
    def _init_senders(self):
        """Initialize message senders for motor commands."""
        self.speedMotorSender = messageHandlerSender(
            self.queuesList, SpeedMotor
        )
        self.steerMotorSender = messageHandlerSender(
            self.queuesList, SteerMotor
        )
        self.motorCommandStatsSender = messageHandlerSender(
            self.queuesList, MotorCommandStats
        )
        self.pidFeedbackSender = messageHandlerSender(
            self.queuesList, PIDFeedback
        )
        self.motorCommandSender = messageHandlerSender(
            self.queuesList, MotorCommand
        )

    # ================================ RUN ================================================
    def thread_work(self):
        """Main motor control loop."""
        try:
            # Get latest inputs
            self._update_inputs()
            
            # Process with PID control
            start_time = time.time()
            self._process_control()
            processing_time = (time.time() - start_time) * 1000  # ms
            self.processing_time = processing_time
            
            # Send motor commands
            self._send_commands()
            self.command_count += 1
            
            # Send feedback periodically
            if self.frame_count % 10 == 0:
                self._send_feedback()
            
            # Send statistics periodically
            if self.frame_count % 20 == 0:
                self._send_statistics()
            
            # Update timing
            self._update_fps()
            self.frame_count += 1
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Motor Controller Error: {e}")

    # ================================ UPDATE INPUTS =========================================
    def _update_inputs(self):
        """Receive latest input data."""
        try:
            # Get autonomous decision
            decision = self.decisionSubscriber.receive()
            if decision is not None:
                self.autonomous_decision = decision
                
                # Update PID setpoints
                self.speed_pid.set_setpoint(decision.get('speed', 0))
                self.steer_pid.set_setpoint(decision.get('steer', 0))
            
            # Get current speed feedback
            speed_msg = self.speedFeedbackSubscriber.receive()
            if speed_msg is not None:
                self.current_speed_feedback = float(speed_msg)
            
            # Get current steering feedback
            steer_msg = self.steerFeedbackSubscriber.receive()
            if steer_msg is not None:
                self.current_steer_feedback = float(steer_msg)
                
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Input update error: {e}")

    # ================================ PROCESS CONTROL =========================================
    def _process_control(self):
        """Main control processing with PID."""
        try:
            # Check for emergency stop
            with self.emergency_stop_lock:
                if self.emergency_stop_active:
                    self.speed_command = 0
                    self.steer_command = 0
                    self.speed_pid.reset()
                    self.steer_pid.reset()
                    return
            
            # If no decision yet, idle
            if self.autonomous_decision is None:
                self.speed_command = 0
                self.steer_command = 0
                return
            
            # Get target values from decision
            target_speed = self.autonomous_decision.get('speed', 0)
            target_steer = self.autonomous_decision.get('steer', 0)
            
            # Update PID controllers
            speed_pid_output = self.speed_pid.update(self.current_speed_feedback)
            steer_pid_output = self.steer_pid.update(self.current_steer_feedback)
            
            # Apply direct command with PID correction
            self.speed_command = self._apply_speed_control(target_speed, speed_pid_output)
            self.steer_command = self._apply_steer_control(target_steer, steer_pid_output)
            
            # Apply smoothing to prevent jerky movements
            self.speed_command = self._smooth_speed(self.speed_command)
            self.steer_command = self._smooth_steer(self.steer_command)
            
            # Enforce limits
            self.speed_command = np.clip(
                self.speed_command,
                self.MOTOR_SPECS['speed']['min'],
                self.MOTOR_SPECS['speed']['max']
            )
            self.steer_command = np.clip(
                self.steer_command,
                self.MOTOR_SPECS['steering']['min'],
                self.MOTOR_SPECS['steering']['max']
            )
            
            # Store history
            self.speed_history.append(self.speed_command)
            self.steer_history.append(self.steer_command)
            
            # Track errors for diagnostics
            speed_error = self.autonomous_decision.get('speed', 0) - self.current_speed_feedback
            steer_error = self.autonomous_decision.get('steer', 0) - self.current_steer_feedback
            self.speed_error_history.append(speed_error)
            self.steer_error_history.append(steer_error)
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Control processing error: {e}")

    # ================================ APPLY SPEED CONTROL =========================================
    def _apply_speed_control(self, target_speed, pid_output):
        """Apply speed control with PID correction."""
        try:
            # Combine target with PID correction
            # Target dominates, PID provides fine-tuning
            combined_speed = target_speed + pid_output * 0.2
            
            return combined_speed
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Speed control error: {e}")
            return 0

    # ================================ APPLY STEER CONTROL =========================================
    def _apply_steer_control(self, target_steer, pid_output):
        """Apply steering control with PID correction."""
        try:
            # Combine target with PID correction
            combined_steer = target_steer + pid_output * 0.15
            
            return combined_steer
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Steer control error: {e}")
            return 0

    # ================================ SMOOTH SPEED =========================================
    def _smooth_speed(self, target_speed):
        """Apply exponential smoothing to speed commands."""
        try:
            # Smooth transition
            smoothed_speed = (
                self.last_speed_command * (1 - self.speed_smoothing_factor) +
                target_speed * self.speed_smoothing_factor
            )
            
            # Enforce ramp rate (prevent sudden changes)
            max_change = self.MOTOR_SPECS['speed']['ramp_rate']
            smoothed_speed = np.clip(
                smoothed_speed,
                self.last_speed_command - max_change,
                self.last_speed_command + max_change
            )
            
            self.last_speed_command = smoothed_speed
            
            return smoothed_speed
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Speed smoothing error: {e}")
            return self.last_speed_command

    # ================================ SMOOTH STEER =========================================
    def _smooth_steer(self, target_steer):
        """Apply exponential smoothing to steering commands."""
        try:
            # Smooth transition
            smoothed_steer = (
                self.last_steer_command * (1 - self.steer_smoothing_factor) +
                target_steer * self.steer_smoothing_factor
            )
            
            # Enforce ramp rate
            max_change = self.MOTOR_SPECS['steering']['ramp_rate']
            smoothed_steer = np.clip(
                smoothed_steer,
                self.last_steer_command - max_change,
                self.last_steer_command + max_change
            )
            
            self.last_steer_command = smoothed_steer
            
            return smoothed_steer
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Steer smoothing error: {e}")
            return self.last_steer_command

    # ================================ SEND COMMANDS =========================================
    def _send_commands(self):
        """Send motor commands to serial handler."""
        try:
            # Send speed command
            self.speedMotorSender.send(str(int(self.speed_command)))
            
            # Send steering command
            self.steerMotorSender.send(str(int(self.steer_command) * 100))  # Convert to format expected
            
            # Send motor command info
            self.motorCommandSender.send({
                'speed': float(self.speed_command),
                'steer': float(self.steer_command),
                'timestamp': time.time()
            })
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Failed to send commands: {e}")

    # ================================ SEND FEEDBACK =========================================
    def _send_feedback(self):
        """Send PID feedback and diagnostics."""
        try:
            avg_speed_error = np.mean(list(self.speed_error_history)) if self.speed_error_history else 0
            avg_steer_error = np.mean(list(self.steer_error_history)) if self.steer_error_history else 0
            
            feedback = {
                'speed_command': float(self.speed_command),
                'speed_feedback': float(self.current_speed_feedback),
                'speed_error': float(avg_speed_error),
                'steer_command': float(self.steer_command),
                'steer_feedback': float(self.current_steer_feedback),
                'steer_error': float(avg_steer_error),
                'emergency_stop': self.emergency_stop_active,
                'timestamp': time.time()
            }
            
            self.pidFeedbackSender.send(feedback)
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Failed to send feedback: {e}")

    # ================================ SEND STATISTICS =========================================
    def _send_statistics(self):
        """Send motor controller statistics."""
        try:
            stats = {
                'frame_count': self.frame_count,
                'command_count': self.command_count,
                'processing_time_ms': self.processing_time,
                'fps': self.fps,
                'speed_command': float(self.speed_command),
                'speed_feedback': float(self.current_speed_feedback),
                'steer_command': float(self.steer_command),
                'steer_feedback': float(self.current_steer_feedback),
                'emergency_stop_active': self.emergency_stop_active,
                'pid_speed_integral': float(self.speed_pid.integral),
                'pid_steer_integral': float(self.steer_pid.integral),
                'timestamp': time.time()
            }
            
            self.motorCommandStatsSender.send(stats)
            
            if self.debugger and self.frame_count % 60 == 0:
                self.logger.info(
                    f"Motor Controller - Speed: {self.speed_command:.1f} mm/s, "
                    f"Steer: {self.steer_command:.1f}°, "
                    f"FPS: {self.fps:.1f}, "
                    f"Time: {self.processing_time:.1f}ms"
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

    # ================================ TRIGGER EMERGENCY STOP =========================================
    def trigger_emergency_stop(self):
        """Trigger emergency stop."""
        try:
            with self.emergency_stop_lock:
                self.emergency_stop_active = True
            
            if self.debugger:
                self.logger.warning("EMERGENCY STOP TRIGGERED IN MOTOR CONTROLLER")
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Emergency stop error: {e}")

    # ================================ RESUME FROM EMERGENCY STOP =========================================
    def resume_from_emergency_stop(self):
        """Resume from emergency stop."""
        try:
            with self.emergency_stop_lock:
                self.emergency_stop_active = False
            
            # Reset PID controllers
            self.speed_pid.reset()
            self.steer_pid.reset()
            
            if self.debugger:
                self.logger.info("Resumed from emergency stop")
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Resume error: {e}")

    # ================================ TUNE PID =========================================
    def tune_pid_speed(self, kp, ki, kd):
        """Tune speed PID parameters online."""
        try:
            self.speed_pid.kp = kp
            self.speed_pid.ki = ki
            self.speed_pid.kd = kd
            
            if self.debugger:
                self.logger.info(f"Speed PID tuned: Kp={kp}, Ki={ki}, Kd={kd}")
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"PID tuning error: {e}")

    def tune_pid_steer(self, kp, ki, kd):
        """Tune steering PID parameters online."""
        try:
            self.steer_pid.kp = kp
            self.steer_pid.ki = ki
            self.steer_pid.kd = kd
            
            if self.debugger:
                self.logger.info(f"Steer PID tuned: Kp={kp}, Ki={ki}, Kd={kd}")
            
        except Exception as e:
            if self.debugger:
                self.logger.error(f"PID tuning error: {e}")

    # ================================ STATE CHANGE HANDLER ========================================
    def state_change_handler(self):
        """Handle state changes from the state machine."""
        message = self.stateChangeSubscriber.receive()
        if message is not None:
            try:
                modeDict = SystemMode[message].value["autonomy"]["thread"]
                
                if self.debugger:
                    self.logger.info(f"Motor Controller mode changed to: {message}")
            except Exception as e:
                if self.debugger:
                    self.logger.error(f"State change error: {e}")

    # ================================ STOP ================================================
    def stop(self):
        """Clean up and stop the thread."""
        try:
            # Send zero commands
            self.speedMotorSender.send("0")
            self.steerMotorSender.send("0")
            
            if self.debugger:
                self.logger.info(
                    f"Motor Controller thread stopped. "
                    f"Sent {self.command_count} commands, "
                    f"Avg FPS: {self.fps:.1f}"
                )
        except Exception as e:
            if self.debugger:
                self.logger.error(f"Error during shutdown: {e}")
        
        super(threadMotorController, self).stop()