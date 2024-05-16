import cv2
import numpy as np
import matplotlib.pyplot as plt
from frame_shifting import frame_shifting_control_algorithm as fsca

class VibrationAnalyzer:
    def __init__(self, frames_directory, video_path):
        self.frames_directory = frames_directory
        self.video_path = video_path
        self.analyzer = fsca.PatternMatching(frames_directory)
        self.frame_rate = self.get_frame_rate()

    def extract_shifts(self):
        shifts, _ = self.analyzer.pattern_matching_on_frames()
        self.shifts_x = [s[0] for s in shifts]
        self.shifts_y = [s[1] for s in shifts]
        return self.shifts_x, self.shifts_y

    def get_frame_rate(self):
        video = cv2.VideoCapture(self.video_path)
        frame_rate = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return frame_rate

    def calculate_vibration(self, shifts):
        # FFT
        fft_result = np.fft.fft(shifts)
        # Getting the frequencies
        freqs = np.fft.fftfreq(len(shifts), d=(1 / self.frame_rate))
        # Getting the amplitudes
        amplitudes = np.abs(fft_result)
        return freqs, amplitudes

    def analyze_vibration(self):
        shifts_x, shifts_y = self.extract_shifts()
        freqs_x, amplitudes_x = self.calculate_vibration(shifts_x)
        freqs_y, amplitudes_y = self.calculate_vibration(shifts_y)
        return {
            "X-Axis": {"frequencies": freqs_x, "amplitudes": amplitudes_x},
            "Y-Axis": {"frequencies": freqs_y, "amplitudes": amplitudes_y}
        }
