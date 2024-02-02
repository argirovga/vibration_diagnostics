import numpy as np
import random
import cmath
from scipy import signal
from deap import base, creator, tools, algorithms
import logging as log
import pygad
import cv2
from scipy.fft import fft2, ifft2, fftshift
from scipy.signal import iirnotch, lfilter
from event_frames.event_frame_generator import EventFrameManager
import matplotlib.pyplot as plt

class EventFilter:
    def __init__(self, _EF_manager: EventFrameManager) -> None:
        self.EF_manager = _EF_manager
        self.ef_frames = np.empty_like(self.EF_manager.event_frames, dtype=complex)

    def generate_filtered_ef_frames(self):
        print("Start generating EF-frames")

        for t in range(len(self.EF_manager.event_frames)):
            self.ef_frames[t] = cv2.fastNlMeansDenoising(self.EF_manager.event_frames[t], None)

        print("EF-frames successfully generated")

    def iteration(self):
        self.generate_filtered_ef_frames()

    

