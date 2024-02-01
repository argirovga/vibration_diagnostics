import numpy as np
import random
from scipy import signal
from deap import base, creator, tools, algorithms

from event_frames.event_frame_generator import EventFrameManager


class EventFilter:
    def __init__(self, _file_path: str) -> None:
        EF_manager = EventFrameManager(_file_path)

        EF_manager.extract_frames()
        EF_manager.convert_to_grayscale()
        EF_manager.create_event_frames()
        self.K = random.randint(0,255)
        self.a = np.ones(self.K)
        self.b = np.ones(self.K)
        
    

