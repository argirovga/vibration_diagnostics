import os, shutil
import cv2
import numpy as np
from matplotlib import pyplot as plt

class EventFrameManager():

    def __init__(self, _file_path: str) -> None:
        self.file_path = _file_path
        self.regular_frames_count = 0
        self.event_frames = []


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
        while(True): 
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
        
        for currentframe in range(1, self.regular_frames_count):
            # frame1 = cv2.imread('raw_data/raw_frames'+ f'/gray_frame_{currentframe - 1}.jpg')
            # frame2 = cv2.imread('raw_data/raw_frames'+ f'/gray_frame_{currentframe}.jpg')

            frame1 = cv2.imread('raw_data/raw_frames' + f'/gray_frame_{currentframe - 1}.jpg', cv2.IMREAD_GRAYSCALE)
            frame2 = cv2.imread('raw_data/raw_frames' + f'/gray_frame_{currentframe}.jpg', cv2.IMREAD_GRAYSCALE)

            frame_diff = self.compute_frames_difference(frame1, frame2)


            # blur = cv2.GaussianBlur(frame_diff,(5,5),0)
            # _, event_frame = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Gausian filtering => otsu optimization

            
            # _, event_frame = cv2.threshold(frame_diff, 0 , 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            event_frame = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,10)
            # self.event_frames.append(event_frame)

            
            

            name = 'raw_data/event_frames/frame_' + str(currentframe - 1) + '.jpg'
            try:
                cv2.imwrite(name, event_frame)
            except:
                print("Error while writing event frame to file")




