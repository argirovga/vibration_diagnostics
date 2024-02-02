import os
import shutil

import cv2
import numpy as np

from event_filter.event_filter_generator import EventFilter
from event_frames.event_frame_generator import EventFrameManager


EF_manager = EventFrameManager("raw_data/high_speed_cam_videos/MotorAmplif_motion_evm_2022-12-23-10-50-19.mp4")

EF = EventFilter(EF_manager)
EF.generate_filtered_ef_frames()

if not os.path.exists('raw_data/event_frames_filtered'):
    os.makedirs('raw_data/event_frames_filtered')
else:
    for filename in os.listdir('raw_data/event_frames_filtered'):
        file_path = os.path.join('raw_data/event_frames_filtered', filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    print("Data directory cleaned \n____________________________\n")


for currentframe in range(0, len(EF.ef_frames)):
    magnitude = np.abs(EF.ef_frames[currentframe])

    # Convert to uint8
    image = np.uint8(magnitude)
    name = 'raw_data/event_frames_filtered/ev_fram_filt' + str(currentframe) + '.jpg'
    cv2.imwrite(name, image)

# for frame in EF.ef_frames:
#     print(frame)