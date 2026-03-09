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

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../../..")

from src.autonomy.threads.threadMotorController import threadMotorController
from src.autonomy.threads.threadSafetyMonitor import threadSafetyMonitor
from src.autonomy.threads.threadPathPlanner import threadPathPlanner
from src.templates.workerprocess import WorkerProcess
from src.autonomy.threads.threadDecisionMaker import threadDecisionMaker
from src.statemachine.stateMachine import StateMachine
from src.statemachine.systemMode import SystemMode
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.allMessages import StateChange


class processAutonomy(WorkerProcess):
    """This process handles autonomous driving decision making.
    
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        ready_event (multiprocessing.Event): Event to signal when process is ready
        debugging (bool, optional): A flag for debugging. Defaults to False.
    """

    # ====================================== INIT ==========================================
    def __init__(self, queueList, logging, ready_event=None, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.stateChangeSubscriber = messageHandlerSubscriber(self.queuesList, StateChange, "lastOnly", True)

        super(processAutonomy, self).__init__(self.queuesList, ready_event)

    # ================================ STATE CHANGE HANDLER ========================================
    def state_change_handler(self):
        """Handle state changes from the state machine."""
        message = self.stateChangeSubscriber.receive()
        if message is not None:
            modeDict = SystemMode[message].value["autonomy"]["process"]

            if modeDict["enabled"] == True:
                self.resume_threads()
            elif modeDict["enabled"] == False:
                self.pause_threads()

    # ===================================== INIT THREADS ======================================
    def _init_threads(self):
    """Create the Autonomy threads and add to the list of threads."""
    
        # Decision maker thread
            pathPlannerTh = threadPathPlanner(
            self.queuesList, self.logging, self.debugging
        )
        self.threads.append(pathPlannerTh)
        
        # Decision maker thread (10 Hz - tactical decisions)
        decisionMakerTh = threadDecisionMaker(
            self.queuesList, self.logging, self.debugging
        )
        self.threads.append(decisionMakerTh)
        
        # Motor controller thread (20 Hz - smooth control)
        motorControllerTh = threadMotorController(
            self.queuesList, self.logging, self.debugging
        )
        self.threads.append(motorControllerTh)
        
        # Safety monitor thread (50 Hz - continuous monitoring)
        safetyMonitorTh = threadSafetyMonitor(
            self.queuesList, self.logging, self.debugging
        )
        self.threads.append(safetyMonitorTh)

# =================================== EXAMPLE =========================================
if __name__ == "__main__":
    from multiprocessing import Queue, Event
    import time
    import logging

    allProcesses = list()

    debugg = True

    queueList = {
        "Critical": Queue(),
        "Warning": Queue(),
        "General": Queue(),
        "Config": Queue(),
        "Log": Queue(),
    }

    logger = logging.getLogger()

    process = processAutonomy(queueList, logger, debugging=debugg)

    process.daemon = True
    process.start()

    time.sleep(5)
    print("Autonomy process started successfully")
    time.sleep(10)

    process.stop()