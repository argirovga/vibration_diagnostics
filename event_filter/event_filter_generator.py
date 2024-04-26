import numpy as np
import cv2
import os
import shutil
from event_frames.event_frame_generator import EventFrameManager

class EventFilter:
    def __init__(self, ef_manager: EventFrameManager, filter_order: int, b_coefficients: np.ndarray, a_coefficients: np.ndarray) -> None:
        self.ef_manager = ef_manager
        self.filter_order = filter_order
        self.b_coefficients = b_coefficients
        self.a_coefficients = a_coefficients
        self.filtered_event_frames = np.zeros_like(self.ef_manager.event_frames)

    def generate_filtered_ef_frames(self, save_frames: bool = False, output_dir: str = None) -> None:
        print("Filtering try")

        self.filtered_event_frames = np.zeros_like(self.ef_manager.event_frames)

        for ind in range(self.ef_manager.event_frames_count - 1):
            new_frame = np.zeros_like(self.ef_manager.event_frames[ind], dtype=np.float64)
            
            # B sum
            for n in range(self.filter_order + 1):
                if ind - n < 0: break
                else:
                    new_frame += self.b_coefficients[n] * self.ef_manager.event_frames[ind - n]
            
            # A sum
            for m in range(1, self.filter_order + 1):
                if ind - m < 0: break
                else:
                    new_frame -= self.a_coefficients[m - 1] * self.ef_manager.event_frames[ind - m]
            self.filtered_event_frames[ind] = new_frame

        print("Filtering try completed")

        if save_frames and output_dir:
            print("Saving the resulting frames")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            else:
                for filename in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_path, e))
                print("Data directory cleaned \n____________________________\n")

            for idx, frame in enumerate(self.filtered_event_frames):
                name = os.path.join(output_dir, f'filtered_frame_{idx}.jpg')
                try:
                    temp_increase = np.empty_like(frame)
                    temp_increase.fill(20)
                    frame += temp_increase
                    cv2.imwrite(name, frame)
                except:
                    print("Error while writing filtered event frame to file")
