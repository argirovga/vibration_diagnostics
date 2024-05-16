import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import cv2
import os
from concurrent.futures import ThreadPoolExecutor


class FrameShiftAnalyzer:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.frames = self.load_frames_from_directory()

    def phase_correlation(self, f1, f2):
        # Using The Hahn window to reduce spectral artifacts (spectrum leakage) that occur when analyzing signals using
        # the Fourier transform method. It smoothes out discontinuities at the edges of the sample so that spectral analysis
        # (e.g. frequency transform) is more accurate and not distorted by unwanted frequency components.
        # Also, perfoming FFT on the frames
        F1 = fft2(f1 * np.hanning(f1.shape[0])[:, None] * np.hanning(f1.shape[1])[None, :])
        F2 = fft2(f2 * np.hanning(f2.shape[0])[:, None] * np.hanning(f2.shape[1])[None, :])

        # Compute the cross-power spectrum
        R = F1 * np.conj(F2)
        R /= np.abs(R + 1e-8) if np.any(R) else 1 # Prevent division by zero by adding a small value

        # Apply inverse Fourier Transform
        shifted = fftshift(ifft2(R))

        # Calculate absolute values
        abs_shifted = np.abs(shifted)

        # Extract all pixel values along the x and y axes
        rows = abs_shifted[:, 0].tolist()  # All pixel values along the y-axis
        cols = abs_shifted[0, :].tolist()  # All pixel values along the x-axis

        return rows, cols

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

        return frames

    def compute_shift(self, frame_pair):
        return self.phase_correlation(frame_pair[0], frame_pair[1])

    def pattern_matching_on_frames(self):
        shifts = []
        # Использование ThreadPoolExecutor для параллелизации вычислений
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_shifts = [executor.submit(self.compute_shift, (self.frames[i], self.frames[i + 1]))
                             for i in range(len(self.frames) - 1)]
            for future in future_shifts:
                shift = future.result()
                shifts.append(shift)
        return shifts



