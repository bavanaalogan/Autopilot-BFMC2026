if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../../..")

from src.templates.workerprocess import WorkerProcess
from src.autonomy.PathPlanner.threads.threadPathPlanner import threadPathPlanner

class processPathPlanner(WorkerProcess):
    """This process handles PathPlanner.
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool, optional): A flag for debugging. Defaults to False.
    """

    def __init__(self, queueList, logging, ready_event=None, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        super(processPathPlanner, self).__init__(self.queuesList, ready_event)

    def state_change_handler(self):
        pass

    def process_work(self):
        pass

    def _init_threads(self):
        """Create the PathPlanner Publisher thread and add to the list of threads."""
        PathPlannerTh = threadPathPlanner(
            self.queuesList, self.logging, self.debugging
        )
        self.threads.append(PathPlannerTh)
