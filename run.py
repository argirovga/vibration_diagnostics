import numpy as np
from deap import base, creator, tools, algorithms
from event_frames.event_frame_generator import EventFrameManager
from event_filter.event_filter_generator import EventFilter

EF = EventFilter("raw_data/high_speed_cam_videos/MotorAmplif_motion_evm_2022-12-23-10-50-19.mp4")

