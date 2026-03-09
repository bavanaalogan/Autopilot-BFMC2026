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
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

from enum import Enum

####################################### processCamera #######################################
class mainCamera(Enum):
    Queue = "General"
    Owner = "threadCamera"
    msgID = 1
    msgType = "str"

class serialCamera(Enum):
    Queue = "General"
    Owner = "threadCamera"
    msgID = 2
    msgType = "str"

class Recording(Enum):
    Queue = "General"
    Owner = "threadCamera"
    msgID = 3
    msgType = "bool"

class Signal(Enum):
    Queue = "General"
    Owner = "threadCamera"
    msgID = 4
    msgType = "str"

class LaneKeeping(Enum):
    Queue = "General"
    Owner = "threadCamera" # here you will send an offset of the car position between the lanes of the road + - from 0 point to dashboard
    msgID = 5
    msgType = "int"

################################# processCarsAndSemaphores ##################################
class Cars(Enum):
    Queue = "General"
    Owner = "threadCarsAndSemaphores"
    msgID = 1
    msgType = "dict"

class Semaphores(Enum):
    Queue = "General"
    Owner = "threadCarsAndSemaphores"
    msgID = 2
    msgType = "dict"

################################# From Dashboard ##################################
class SpeedMotor(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 1
    msgType = "str"

class SteerMotor(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 2
    msgType = "str"

class Control(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 3
    msgType = "dict"

class Brake(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 4
    msgType = "str"

class Record(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 5
    msgType = "str"

class Config(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 6
    msgType = "dict"

class Klem(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 7
    msgType = "str"

class DrivingMode(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 8
    msgType = "str"

class ToggleInstant(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 9
    msgType = "str"

class ToggleBatteryLvl(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 10
    msgType = "str"

class ToggleImuData(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 11
    msgType = "str"

class ToggleResourceMonitor(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 12
    msgType = "str"

class State(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 13
    msgType = "str"

class Brightness(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 14
    msgType = "str"

class Contrast(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 15
    msgType = "str"

class DropdownChannelExample(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 16
    msgType = "str"

class SliderChannelExample(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 17
    msgType = "str"

class ControlCalib(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 18
    msgType = "dict"

class IsAlive(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 19
    msgType = "bool"

class RequestSteerLimits(Enum):
    Queue = "General"
    Owner = "Dashboard"
    msgID = 20
    msgType = "bool"


################################# From Nucleo ##################################
class BatteryLvl(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 1
    msgType = "int"

class ImuData(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 2
    msgType = "str"

class InstantConsumption(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 3
    msgType = "float"

class ResourceMonitor(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 4
    msgType = "dict"

class CurrentSpeed(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 5
    msgType = "float"

class CurrentSteer(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 6
    msgType = "float"

class ImuAck(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 7
    msgType = "str"
    
class ShutDownSignal(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 8
    msgType = "str"

class CalibPWMData(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 9
    msgType = "dict"

class CalibRunDone(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 10
    msgType = "bool"

class AliveSignal(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 11
    msgType = "bool"

class SteeringLimits(Enum):
    Queue = "General"
    Owner = "threadRead"
    msgID = 12
    msgType = "dict"


################################# From Locsys ##################################
class Location(Enum):
    Queue = "General"
    Owner = "threadTrafficCommunication"
    msgID = 1
    msgType = "dict"

######################    From processSerialHandler  ###########################
class EnableButton(Enum):
    Queue = "General"
    Owner = "threadWrite"
    msgID = 1
    msgType = "bool"

class WarningSignal(Enum):
    Queue = "General"
    Owner = "brain"
    msgID = 2
    msgType = "str"

class SerialConnectionState(Enum):
    Queue = "General"
    Owner = "processSerialHandler"
    msgID = 3
    msgType = "bool"

################################# From StateMachine ##################################
class StateChange(Enum):
    Queue = "Critical"
    Owner = "stateMachine"
    msgID = 1
    msgType = "str"

### It will have this format: {"WarningName":"name1", "WarningID": 1}
################################# From Vision ##################################
class LaneDetection(Enum):
    Queue = "General"
    Owner = "threadLaneDetection"
    msgID = 50
    msgType = "dict"

class LanePositionError(Enum):
    Queue = "General"
    Owner = "threadLaneDetection"
    msgID = 51
    msgType = "float"

class LaneConfidence(Enum):
    Queue = "General"
    Owner = "threadLaneDetection"
    msgID = 52
    msgType = "float"

class ProcessedFrame(Enum):
    Queue = "General"
    Owner = "threadLaneDetection"
    msgID = 53
    msgType = "str"  # base64 encoded image for dashboard

class ObjectDetection(Enum):
    Queue = "General"
    Owner = "threadObjectDetection"
    msgID = 54
    msgType = "dict"

class ObstacleWarning(Enum):
    Queue = "Warning"
    Owner = "threadObjectDetection"
    msgID = 55
    msgType = "dict"

class DetectionFPS(Enum):
    Queue = "General"
    Owner = "threadObjectDetection"
    msgID = 56
    msgType = "float"

class DetectionStats(Enum):
    Queue = "General"
    Owner = "threadObjectDetection"
    msgID = 57
    msgType = "dict"

class SignDetection(Enum):
    Queue = "General"
    Owner = "threadSignDetection"
    msgID = 58
    msgType = "dict"

class SignPosition(Enum):
    Queue = "General"
    Owner = "threadSignDetection"
    msgID = 59
    msgType = "dict"

class SignConfidence(Enum):
    Queue = "General"
    Owner = "threadSignDetection"
    msgID = 60
    msgType = "float"

class TrafficSignStats(Enum):
    Queue = "General"
    Owner = "threadSignDetection"
    msgID = 61
    msgType = "dict"

class ProcessedFrame(Enum):
    Queue = "General"
    Owner = "threadFramePreprocessor"
    msgID = 62
    msgType = "str"  # base64 encoded numpy array

class FrameBuffer(Enum):
    Queue = "General"
    Owner = "threadFramePreprocessor"
    msgID = 63
    msgType = "dict"

class FramePreprocessingStats(Enum):
    Queue = "General"
    Owner = "threadFramePreprocessor"
    msgID = 64
    msgType = "dict"

class LightingCondition(Enum):
    Queue = "General"
    Owner = "threadFramePreprocessor"
    msgID = 65
    msgType = "str"

################################# From Autonomy ##################################
class AutonomousDecision(Enum):
    Queue = "General"
    Owner = "threadDecisionMaker"
    msgID = 70
    msgType = "dict"

class EmergencyStop(Enum):
    Queue = "Critical"
    Owner = "threadDecisionMaker"
    msgID = 71
    msgType = "dict"

class DecisionMakerStats(Enum):
    Queue = "General"
    Owner = "threadDecisionMaker"
    msgID = 72
    msgType = "dict"

class AutonomyState(Enum):
    Queue = "General"
    Owner = "threadDecisionMaker"
    msgID = 73
    msgType = "str"

class DecisionLog(Enum):
    Queue = "General"
    Owner = "threadDecisionMaker"
    msgID = 74
    msgType = "dict"

class MotorCommandStats(Enum):
    Queue = "General"
    Owner = "threadMotorController"
    msgID = 75
    msgType = "dict"

class PIDFeedback(Enum):
    Queue = "General"
    Owner = "threadMotorController"
    msgID = 76
    msgType = "dict"

class MotorCommand(Enum):
    Queue = "General"
    Owner = "threadMotorController"
    msgID = 77
    msgType = "dict"

class PlannedPath(Enum):
    Queue = "General"
    Owner = "threadPathPlanner"
    msgID = 78
    msgType = "dict"

class Waypoint(Enum):
    Queue = "General"
    Owner = "threadPathPlanner"
    msgID = 79
    msgType = "dict"

class SafetyAlert(Enum):
    Queue = "Warning"
    Owner = "threadSafetyMonitor"
    msgID = 80
    msgType = "dict"

class SensorHealth(Enum):
    Queue = "General"
    Owner = "threadSafetyMonitor"
    msgID = 81
    msgType = "dict"

class CommandValidation(Enum):
    Queue = "General"
    Owner = "threadSafetyMonitor"
    msgID = 82
    msgType = "dict"