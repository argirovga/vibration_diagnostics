import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import cv2
import os
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor


class PatternMatching:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.frames = self.load_frames_from_directory()

    def phase_correlation(self, f1, f2):
        # Using The Hahn window to reduce spectral artifacts (spectrum leakage) that occur when analyzing signals using
        # the Fourier transform method. It smoothes out discontinuities at the edges of the sample so that spectral analysis
        # (e.g. frequency transform) is more accurate and not distorted by unwanted frequency components.
        # Also perfoming FFT on the frames
        F1 = fft2(f1 * np.hanning(f1.shape[0])[:, None] * np.hanning(f1.shape[1])[None, :])
        F2 = fft2(f2 * np.hanning(f2.shape[0])[:, None] * np.hanning(f2.shape[1])[None, :])


        # Compute the cross-power spectrum
        R = F1 * np.conj(F2)
        R /= np.abs(R) if np.any(R) else 1

        # Apply inverse Fourier Transform
        shifted = fftshift(ifft2(R))

        # Calculate absolute values
        abs_shifted = np.abs(shifted)

        # Create indexes for each axis
        rows, cols = np.indices(abs_shifted.shape)

        # Calculate weighted averages (center of mass)
        mean_x = np.sum(cols * abs_shifted) / np.sum(abs_shifted)
        mean_y = np.sum(rows * abs_shifted) / np.sum(abs_shifted)

        mean_y, mean_x = int(round(mean_y)), int(round(mean_x))

        return mean_x, mean_y

    def load_frames_from_directory(self):
        def read_frame(frame_name):
            frame_path = os.path.join(self.directory_path, frame_name)
            return cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

        # Get all the file names in sorted order
        frame_names = sorted(os.listdir(self.directory_path))

        # Use ThreadPoolExecutor to read frames in parallel
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Считываем все кадры параллельно и формируем список результатов
            frames = list(executor.map(read_frame, frame_names))

        # Remove frames that have not been successfully read (None)
        frames = [frame for frame in frames if frame is not None]

        return frames

    def compute_shift(self, frame_pair):
        return self.phase_correlation(frame_pair[0], frame_pair[1])

    def pattern_matching_on_frames(self):
        timings = []
        shifts = []
        # Using ThreadPoolExecutor to parallelize the computation
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_shifts = [executor.submit(self.compute_shift, (self.frames[i], self.frames[i + 1]))
                             for i in range(len(self.frames) - 1)]
            for future in future_shifts:
                start_time = time.time()
                shift = future.result()
                shifts.append(shift)
                end_time = time.time()
                timings.append(end_time - start_time)
        return shifts, timings


if __name__ == "__main__":
    start_time_script = time.time()  # Record the start time of the script
    frames_directory = "C:\\Users\\1\\Desktop\\pythonProject\\vibration_diagnostics_copy\\raw_data\\event_frames_filtered"
    analyzer = PatternMatching(frames_directory)
    shifts, timings = analyzer.pattern_matching_on_frames()
    plt.plot(timings, marker='o')
    plt.title('Time Taken for Each Phase Correlation Calculation')
    plt.xlabel('Frame Pair Index')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.show()
    end_time_script = time.time()  # Record the end time of the script
    total_time_script = end_time_script - start_time_script
    print(f"Total script execution time: {total_time_script} seconds")
