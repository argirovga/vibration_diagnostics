import os, shutil
import cv2
import numpy as np
from matplotlib import pyplot as plt


class EventFrameManager():

    def __init__(self, _file_path: str) -> None:
        self.file_path = _file_path
        self.regular_frames_count = 0
        self.event_frames = []
        self.best_c_constant = None
        self.best_block_size = None

    def extract_frames(self):
        cam = cv2.VideoCapture(self.file_path)

        try:
            if not os.path.exists('raw_data/raw_frames'):
                os.makedirs('raw_data/raw_frames')
            else:
                for filename in os.listdir('raw_data/raw_frames'):
                    file_path = os.path.join('raw_data/raw_frames', filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_path, e))
                print("Data directory cleaned \n____________________________\n")
        except OSError:
            raise ValueError("Error while creating new directory")

        currentframe = 0
        while (True):
            frames_left_flag, frame = cam.read()

            if frames_left_flag:
                name = './raw_data/raw_frames/frame_' + str(currentframe) + '.jpg'
                cv2.imwrite(name, frame)

                currentframe += 1
            else:
                print(f'{currentframe} frames created in \"raw_data/raw_frames\" directory \n')
                self.regular_frames_count = currentframe
                break

        cam.release()
        cv2.destroyAllWindows()

    def convert_to_grayscale(self):
        currentframe = 0
        for filename in os.listdir('raw_data/raw_frames'):
            file_path = os.path.join('raw_data/raw_frames', filename)

            frame = cv2.imread(file_path)
            os.unlink(file_path)
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            name = 'raw_data/raw_frames/gray_frame_' + str(currentframe) + '.jpg'
            cv2.imwrite(name, gray_image)

            currentframe += 1

        print(f'{currentframe} frames converted to gray_scale in \"raw_data/raw_frames\" directory \n')

        cv2.destroyAllWindows()

    def compute_frames_difference(self, prev_frame, next_frame):
        return cv2.absdiff(prev_frame, next_frame)

    def automated_threshold_search(self, block_size_range, c_constant_range):
        """
        Search for optimal block size and C constant for adaptive thresholding.
        image_paths: List of paths to images to be used for thresholding.
        block_size_range: Tuple (min, max, step) for block size.
        c_constant_range: Tuple (min, max, step) for C constant.
        """

        best_block_size = None
        best_c_constant = None
        best_score = float('inf')  # You can change the criteria based on your requirement

        for block_size in range(*block_size_range):
            if block_size % 2 == 0:  # Ensure block_size is odd
                continue

            for c_constant in range(*c_constant_range):
                # Cumulative score for the current combination of parameters
                cumulative_score = 0

                # for path in self:
                #     # Read image in grayscale
                #     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                #
                #     # Apply adaptive threshold
                #     thresh_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                #                                        cv2.THRESH_BINARY, block_size, c_constant)

                for currentframe in range(1, self.regular_frames_count):
                    frame1 = cv2.imread('raw_data/raw_frames' + f'/gray_frame_{currentframe - 1}.jpg',
                                        cv2.IMREAD_GRAYSCALE)
                    frame2 = cv2.imread('raw_data/raw_frames' + f'/gray_frame_{currentframe}.jpg', cv2.IMREAD_GRAYSCALE)

                    frame_diff = self.compute_frames_difference(frame1, frame2)
                    thresh_img = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY, block_size, c_constant)

                    # Calculate the score for this thresholded image
                    # For simplicity, let's use the number of white pixels as a score
                    # This can be replaced with any other criterion relevant to your application
                    score = np.sum(thresh_img == 255)
                    cumulative_score += score

                # Update the best parameters if the current score is better
                if cumulative_score < best_score:
                    best_score = cumulative_score
                    best_block_size = block_size
                    best_c_constant = c_constant
                    print(f'Best score: {best_score}, Best block size: {best_block_size}, Best C constant: {best_c_constant}')
        self.best_block_size = best_block_size
        self.best_c_constant = best_c_constant
        return best_block_size, best_c_constant

    def create_event_frames(self):
        if not os.path.exists('raw_data/event_frames'):
            os.makedirs('raw_data/event_frames')
        else:
            for filename in os.listdir('raw_data/event_frames'):
                file_path = os.path.join('raw_data/event_frames', filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            print("Data directory cleaned \n____________________________\n")
        self.automated_threshold_search((3, 15, 2), (-10, 10, 2))

        for currentframe in range(1, self.regular_frames_count):
            # frame1 = cv2.imread('raw_data/raw_frames'+ f'/gray_frame_{currentframe - 1}.jpg')
            # frame2 = cv2.imread('raw_data/raw_frames'+ f'/gray_frame_{currentframe}.jpg')

            frame1 = cv2.imread('raw_data/raw_frames' + f'/gray_frame_{currentframe - 1}.jpg', cv2.IMREAD_GRAYSCALE)
            frame2 = cv2.imread('raw_data/raw_frames' + f'/gray_frame_{currentframe}.jpg', cv2.IMREAD_GRAYSCALE)

            frame_diff = self.compute_frames_difference(frame1, frame2)

            # blur = cv2.GaussianBlur(frame_diff,(5,5),0)
            # _, event_frame = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Gausian filtering => otsu optimization

            # _, event_frame = cv2.threshold(frame_diff, 0 , 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # _, event_frame = cv2.threshold(frame_diff, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            event_frame = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                self.best_block_size, self.best_c_constant)
            # self.event_frames.append(event_frame)

            name = 'raw_data/event_frames/frame_' + str(currentframe - 1) + '.jpg'
            try:
                cv2.imwrite(name, event_frame)
            except:
                print("Error while writing event frame to file")
