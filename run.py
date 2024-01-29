
from event_frames.event_frame_generator import EventFrameManager



EF_manager = EventFrameManager("raw_data/high_speed_cam_videos/MotorAmplif_motion_evm_2022-12-23-10-50-19.mp4")

# Extracting images from video
EF_manager.extract_frames()

# Applying grayscale filter to every frame
EF_manager.convert_to_grayscale()

# Creating event frames
EF_manager.create_event_frames()