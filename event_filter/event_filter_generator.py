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

    def apply_iir_filter(self, frame_t: np.ndarray, prev_output: np.ndarray) -> np.ndarray:
        filtered_output = np.zeros_like(frame_t, dtype=np.float64)

        for n in range(self.filter_order + 1):
            if n == 0:
                filtered_output += self.b_coefficients[n] * frame_t
            else:
                filtered_output -= self.a_coefficients[n] * prev_output

        return filtered_output

    def generate_filtered_ef_frames(self, save_frames: bool = False, output_dir: str = None) -> None:
        print("Filtering try")

        prev_output = np.zeros_like(self.ef_manager.event_frames[0], dtype=np.float64)

        self.filtered_ef_frames = []
        for frame_t in self.ef_manager.event_frames:
            filtered_frame = self.apply_iir_filter(frame_t.astype(np.float64), prev_output)
            self.filtered_ef_frames.append(filtered_frame)
            prev_output = filtered_frame

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

            for idx, frame in enumerate(self.filtered_ef_frames):
                name = os.path.join(output_dir, f'filtered_frame_{idx}.jpg')
                try:
                    cv2.imwrite(name, frame)
                except:
                    print("Error while writing filtered event frame to file")
