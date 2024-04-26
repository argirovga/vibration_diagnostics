import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import cv2
import os

class PatternMatching():
    def __init__(self, _frames: list) -> None:
        self.frames = _frames
        self.shifts = []
        self.pattern_matching_on_frames()

    def phase_correlation(self, f1, f2):
        # Apply Fourier Transform to both frames
        F1 = fft2(f1)
        F2 = fft2(f2)
        # Compute the cross-power spectrum
        R = F1 * np.conj(F2)
        R /= np.abs(R)
        # Apply inverse Fourier Transform
        shifted = fftshift(ifft2(R))
        # Find the peak location, indicating the shift
        # Здесь просто находим максимальное значение в массиве и его индекс (координаты) - это и есть сдвиг
        # np.argmax() - возвращает индекс максимального значения в массиве
        # np.unravel_index() - преобразует индекс в координаты
        shift_y, shift_x = np.unravel_index(np.argmax(np.abs(shifted)), shifted.shape)
        return shift_x, shift_y

    def pattern_matching_on_frames(self):
        for i in range(len(self.frames) - 1):
            shift = self.phase_correlation(self.frames[i], self.frames[i + 1])
            self.shifts.append(shift)

    def get_shifts(self):
        # Вывод формата первый кадр - второй кадр: сдвиг по x, сдвиг по y(т.е. насколько надо сдвинуть первый кадр, чтобы он совпал с вторым по осям)
        return self.shifts

    @staticmethod
    def load_frames_from_directory(directory_path):
        frames = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            frame = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            frames.append(frame)
        return frames

    @staticmethod
    def start():
        frames_directory = "C:\\Users\\1\\Desktop\\pythonProject\\vibration_diagnostics_copy\\raw_data\\event_frames_filtered"
        frames = PatternMatching.load_frames_from_directory(frames_directory)
        shifts = PatternMatching.pattern_matching_on_frames(frames)
        # Вывод формата первый кадр - второй кадр: сдвиг по x, сдвиг по y(т.е. насколько надо сдвинуть первый кадр, чтобы он совпал с вторым по осям)
        for i, shift in enumerate(shifts):
            print(
                f"to align the {i} frame with the {i + 1} frame, we need to move {i} frame: shift x = {shift[0]}, shift y = {shift[1]}")

if __name__ == "__main__":
    PatternMatching.start()

