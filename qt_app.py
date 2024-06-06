import sys
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout,
                             QWidget, QAction, QFrame, QHBoxLayout, QSlider, QStyle, QSizePolicy,
                             QSplitter, QStackedWidget)
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import Qt, QUrl, QTimer
from PyQt5.QtGui import QImage, QPixmap

from event_frames.event_frame_generator import EventFrameManager
from event_filter.event_filter_generator import EventFilter
from event_filter.genetic_algorithm import GeneticAlgorithmCreator
from vibration_calculation.vibration_calculation_algorithm import VibrationAnalyzer

class VideoPlayerApp(QMainWindow):
    def __init__(self):
        super(VideoPlayerApp, self).__init__()

        self.setWindowTitle("VibroDiagnostic")
        self.setGeometry(100, 100, 800, 1200)  

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_widget.setMinimumSize(700, 800)

        self.vibration_frames_directory = 'raw_data/event_frames'

        self.splitter = QSplitter(Qt.Vertical)

        # Top part: original video
        self.original_video_frame = QFrame(self.splitter)
        self.original_video_frame.setFrameShape(QFrame.StyledPanel)
        self.original_video_frame.setMinimumSize(850, 300)
        self.original_video_frame.setFixedHeight(380) 

        self.original_video_layout = QVBoxLayout(self.original_video_frame)
        self.original_video_layout.setAlignment(Qt.AlignTop)

        self.original_video_widget = QVideoWidget(self.original_video_frame)
        self.original_video_widget.setMinimumSize(650, 300)
        self.original_video_widget.setFixedHeight(300) 

        self.original_video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.original_video_layout.addWidget(self.original_video_widget)

        # Middle part: switchable content (result video, first graph, second graph)
        self.switchable_widget = QStackedWidget(self.splitter)
        self.switchable_widget.setMinimumSize(650, 300)

        # Result video area
        self.vibration_video_frame = QFrame()
        self.vibration_video_layout = QVBoxLayout(self.vibration_video_frame)
        self.vibration_video_layout.setAlignment(Qt.AlignTop)

        self.vibration_video_widget = QLabel(self.vibration_video_frame)
        self.vibration_video_widget.setMinimumSize(850, 300)
    
        self.vibration_video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vibration_video_widget.setAttribute(Qt.WA_TranslucentBackground)
        self.vibration_video_widget.setAlignment(Qt.AlignCenter)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(0)
        self.threshold_slider.valueChanged.connect(self.update_threshold)

        self.threshold_label = QLabel(f"Threshold: {self.threshold_slider.value()}")

        self.threshold_layout = QHBoxLayout()
        self.threshold_layout.addWidget(QLabel("Threshold:"))
        self.threshold_layout.addWidget(self.threshold_slider)
        self.threshold_layout.addWidget(self.threshold_label)

        self.vibration_video_layout.addWidget(self.vibration_video_widget)
        self.vibration_video_layout.addLayout(self.threshold_layout)

        # Graph 1 area
        self.graph1_frame = QFrame()
        self.graph1_layout = QVBoxLayout(self.graph1_frame)
        self.graph1_widget = QLabel(self.graph1_frame)
        self.graph1_widget.setMinimumSize(650, 300)
        self.graph1_widget.setAlignment(Qt.AlignCenter)
        self.graph1_layout.addWidget(self.graph1_widget)

        # Graph 2 area
        self.graph2_frame = QFrame()
        self.graph2_layout = QVBoxLayout(self.graph2_frame)
        self.graph2_widget = QLabel(self.graph2_frame)
        self.graph2_widget.setMinimumSize(650, 300)
        self.graph2_widget.setAlignment(Qt.AlignCenter)
        self.graph2_layout.addWidget(self.graph2_widget)

        # Add widgets to switchable area
        self.switchable_widget.addWidget(self.vibration_video_frame)
        self.switchable_widget.addWidget(self.graph1_frame)
        self.switchable_widget.addWidget(self.graph2_frame)

        # Add the original video and switchable widget to the splitter
        self.splitter.addWidget(self.original_video_frame)
        self.splitter.addWidget(self.switchable_widget)

        # Control buttons for switching views
        self.switch_to_video_button = QPushButton("Show Result Video")
        self.switch_to_graph1_button = QPushButton("Show X-Axis Graph")
        self.switch_to_graph2_button = QPushButton("Show Y-Axis Graph")

        self.switch_to_video_button.clicked.connect(lambda: self.switch_view(0))
        self.switch_to_graph1_button.clicked.connect(lambda: self.switch_view(1))
        self.switch_to_graph2_button.clicked.connect(lambda: self.switch_view(2))

        self.switch_layout = QHBoxLayout()
        self.switch_layout.addWidget(self.switch_to_video_button)
        self.switch_layout.addWidget(self.switch_to_graph1_button)
        self.switch_layout.addWidget(self.switch_to_graph2_button)

        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.addLayout(self.switch_layout)
        self.main_layout.addWidget(self.splitter)

        self.btn_play = QPushButton(self.style().standardIcon(QStyle.SP_MediaPlay), "Play", self.original_video_frame)
        self.btn_stop = QPushButton(self.style().standardIcon(QStyle.SP_MediaStop), "Stop", self.original_video_frame)
        self.btn_delete = QPushButton(self.style().standardIcon(QStyle.SP_DialogDiscardButton), "Delete", self.original_video_frame)

        self.position_slider = QSlider(Qt.Horizontal, self.original_video_frame)
        self.position_slider.setRange(0, 0)

        self.label_status = QLabel(self.original_video_frame)
        self.label_status.setText("No video loaded")

        self.control_layout = QHBoxLayout()
        self.control_layout.addWidget(self.btn_play)
        self.control_layout.addWidget(self.btn_stop)
        self.control_layout.addWidget(self.btn_delete)
        self.control_layout.addWidget(self.position_slider)
        self.control_layout.addWidget(self.label_status)

        self.original_video_layout.addLayout(self.control_layout)

        self.btn_play.clicked.connect(self.play_video)
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_delete.clicked.connect(self.delete_video)
        self.position_slider.sliderMoved.connect(self.setPosition)

        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.original_video_widget)
        self.media_player.stateChanged.connect(self.mediaStateChanged)
        self.media_player.positionChanged.connect(self.positionChanged)
        self.media_player.durationChanged.connect(self.durationChanged)
        self.media_player.error.connect(self.handleError)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.vibration_data = []
        self.frame_idx = 0
        self.threshold = 0

        self.open_action = QAction('&Open', self)
        self.open_action.triggered.connect(self.open_file)
        self.save_action = QAction('&Save Results', self)
        self.save_action.triggered.connect(self.save_results)

        self.menu_bar = self.menuBar()
        self.file_menu = self.menu_bar.addMenu('&File')
        self.file_menu.addAction(self.open_action)
        self.file_menu.addAction(self.save_action)

    def switch_view(self, index):
        self.switchable_widget.setCurrentIndex(index)

    def open_file(self):
        options = QFileDialog.Options()
        video_file, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov)", options=options)
        if video_file:
            self.load_video(video_file)
            self.load_vibration_frames(self.vibration_frames_directory, video_file)

    def load_video(self, file_path):
        self.video_path = file_path
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
        self.label_status.setText(f"Loaded: {file_path}")
        self.position_slider.setVisible(True)

    def plot_vibration(self, freqs, amplitudes, axis, graph_path):
        plt.figure()
        plt.plot(freqs[:len(freqs) // 2],
             amplitudes[:len(amplitudes) // 2])
        plt.title(f"Vibration in {axis} axis")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.savefig(graph_path)
        plt.close()

    def load_vibration_frames(self, directory_path, file_path):
        result = self.start_algorithm(file_path)
        self.plot_vibration(result['X-Axis']['frequencies'], result['X-Axis']['amplitudes'], "X", "x_axis_graph.png")
        self.plot_vibration(result['Y-Axis']['frequencies'], result['Y-Axis']['amplitudes'], "Y", "y_axis_graph.png")
        self.load_graphs()

        frame_files = sorted([os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.png') or f.endswith('.jpg')])
        self.vibration_frames = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in frame_files]
        self.vibration_data = self.vibration_frames
        self.frame_idx = 0

    def load_graphs(self):
        self.load_graph_image("x_axis_graph.png", self.graph1_widget)
        self.load_graph_image("y_axis_graph.png", self.graph2_widget)

    def load_graph_image(self, graph_path, graph_widget):
        pixmap = QPixmap(graph_path)
        graph_widget.setPixmap(pixmap)
        graph_widget.setAlignment(Qt.AlignTop | Qt.AlignHCenter)


    def play_video(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            self.media_player.play()
            self.timer.start(1000 // 24)
            self.btn_play.setText("Pause")

    def stop_video(self):
        self.media_player.stop()
        self.timer.stop()
        self.btn_play.setText("Play")
        self.frame_idx = 0

    def delete_video(self):
        self.media_player.setMedia(QMediaContent())
        self.label_status.clear()
        self.position_slider.setVisible(False)
        self.timer.stop()
        self.frame_idx = 0

    def setPosition(self, position):
        self.media_player.setPosition(position)

    def mediaStateChanged(self, state):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.position_slider.setValue(position)

    def durationChanged(self, duration):
        self.position_slider.setRange(0, duration)

    def handleError(self):
        self.btn_play.setEnabled(False)
        self.label_status.setText("Error: " + self.media_player.errorString())

    def save_results(self):
        save_directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save Results")
        if not save_directory:
            return  

        self.save_result_video(save_directory)

        self.save_graphs(save_directory)

        print("Results saved!")

    def save_result_video(self, save_directory):
        result_video_path = os.path.join(save_directory, "result_video.mp4")
        height, width = self.vibration_frames[0].shape

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(result_video_path, fourcc, 24.0, (width, height))

        for frame in self.vibration_frames:
            color_frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            out.write(color_frame)

        out.release()

    def save_graphs(self, save_directory):
        graph1_path = os.path.join(save_directory, "x_axis_graph.png")
        graph2_path = os.path.join(save_directory, "y_axis_graph.png")

        if os.path.exists("x_axis_graph.png"):
            os.rename("x_axis_graph.png", graph1_path)
        if os.path.exists("y_axis_graph.png"):
            os.rename("y_axis_graph.png", graph2_path)

    def update_frame(self):
        if self.frame_idx >= len(self.vibration_data):
            self.timer.stop()
            return

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        ret, frame = cap.read()
        if not ret:
            self.timer.stop()
            return

        vibrated_frame = self.overlay_vibration_data(frame)
        self.display_frame(vibrated_frame)

        self.frame_idx += 1
        cap.release()

    def overlay_vibration_data(self, frame):
        # Create an overlay image with the same size as the frame, initially black
        overlay = np.zeros_like(frame, dtype=np.uint8)

        # Apply the colormap to the vibration data
        color_mapped_vibration = cv2.applyColorMap(self.vibration_data[self.frame_idx], cv2.COLORMAP_HOT)

        # Only keep the parts where vibration data is above the threshold
        mask = self.vibration_data[self.frame_idx] >= self.threshold
        overlay[mask] = color_mapped_vibration[mask]

        # Blend the overlay with the original frame
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def display_frame(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.vibration_video_widget.setPixmap(pixmap)

    def update_threshold(self, value):
        self.threshold = value
        self.threshold_label.setText(f"Threshold: {self.threshold}")

    def start_algorithm(self, file_path):
        # Initialize the EventFrameManager to extract frames from the video
        EF_manager = EventFrameManager(file_path)

        VA = VibrationAnalyzer("preped_data/filtered_event_frames", file_path)
        result = VA.analyze_vibration()
        return result

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = VideoPlayerApp()
    main_window.show()
    sys.exit(app.exec_())
