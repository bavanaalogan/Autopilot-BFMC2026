if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../../..")

from src.templates.workerprocess import WorkerProcess
from src.vision.threads.threadLaneDetection import threadLaneDetection
from src.statemachine.stateMachine import StateMachine
from src.statemachine.systemMode import SystemMode
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.allMessages import StateChange
from src.vision.threads.threadObjectDetection import threadObjectDetection
from src.vision.threads.threadFramePreprocessor import threadFramePreprocessor
from src.vision.threads.threadSignDetection import threadSignDetection


class processVision(WorkerProcess):
    """This process handles vision-based autonomous driving operations.
    
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

        super(processVision, self).__init__(self.queuesList, ready_event)

    # ================================ STATE CHANGE HANDLER ========================================
    def state_change_handler(self):
        """Handle state changes from the state machine."""
        message = self.stateChangeSubscriber.receive()
        if message is not None:
            modeDict = SystemMode[message].value["vision"]["process"]

            if modeDict["enabled"] == True:
                self.resume_threads()
            elif modeDict["enabled"] == False:
                self.pause_threads()

    # ===================================== INIT THREADS ======================================
    def _init_threads(self):
    """Create the Vision threads and add to the list of threads."""
    
        # Add frame preprocessor thread FIRST (processes raw camera frames)
        framePreprocessorTh = threadFramePreprocessor(
            self.queuesList, self.logging, self.debugging
        )
        self.threads.append(framePreprocessorTh)
        
        # Lane detection thread
        laneDetectionTh = threadLaneDetection(
            self.queuesList, self.logging, self.debugging
        )
        self.threads.append(laneDetectionTh)
        
        # Object detection thread
        objectDetectionTh = threadObjectDetection(
            self.queuesList, self.logging, self.debugging
        )
        self.threads.append(objectDetectionTh)
        
        # Sign detection thread
        signDetectionTh = threadSignDetection(
            self.queuesList, self.logging, self.debugging
        )
        self.threads.append(signDetectionTh)

# =================================== EXAMPLE =========================================
#             ++    THIS WILL RUN ONLY IF YOU RUN THE CODE FROM HERE  ++
#                  in terminal:    python3 processVision.py
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

    process = processVision(queueList, logger, debugging=debugg)

    process.daemon = True
    process.start()

    time.sleep(5)
    print("Vision process started successfully")
    time.sleep(10)

    process.stop()