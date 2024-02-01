import numpy as np
from deap import base, creator, tools, algorithms
from event_frames.event_frame_generator import EventFrameManager
from event_filter.event_filter_generator import EventFilter


EF_manager = EventFrameManager("raw_data/high_speed_cam_videos/MotorAmplif_motion_evm_2022-12-23-10-50-19.mp4")
EF_manager.extract_frames()
EF_manager.convert_to_grayscale()
EF_manager.create_event_frames()


EF = EventFilter(EF_manager)
EF.iteration()
