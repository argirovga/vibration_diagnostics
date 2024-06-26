import os, shutil
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pywt
from scipy.signal import deconvolve

class EventFrameManager():
    def __init__(self, _file_path: str) -> None:
        self.file_path = _file_path
        self.regular_frames_count = 0
        self.event_frames_count = 0
        self.event_frames = []
        self.best_c_constant = None
        self.best_block_size = None
        self.extract_frames()
        self.convert_to_grayscale()
        self.create_event_frames()

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
                name = 'raw_data/raw_frames/frame_' + str(currentframe) + '.jpg'
                cv2.imwrite(name, frame)

                currentframe += 1
            else:
                print(f'{currentframe} frames created in \"raw_data/raw_frames\" directory \n')
                self.regular_frames_count = currentframe
                break

        cam.release()
        cv2.destroyAllWindows()

    def denoise_bilateral(self, frame): # the most efficient filter (statistics in the provided python notebook)
        return cv2.bilateralFilter(frame, 9, 75, 75)
    
    def denoise_non_local_means(self, frame):
        return cv2.fastNlMeansDenoising(frame, None, 30, 7, 21)

    def convert_to_grayscale(self):
        currentframe = 0
        for filename in os.listdir('raw_data/raw_frames'):
            file_path = os.path.join('raw_data/raw_frames', filename)

            frame = cv2.imread(file_path)
            os.unlink(file_path)
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_image = self.denoise_bilateral(gray_image)

            name = 'raw_data/raw_frames/gray_frame_' + str(currentframe) + '.jpg'
            cv2.imwrite(name, gray_image)

            currentframe += 1

        print(f'{currentframe} frames converted to gray_scale in \"raw_data/raw_frames\" directory \n')

        cv2.destroyAllWindows()

    def compute_frames_difference(self, prev_frame, next_frame):
        return cv2.absdiff(prev_frame, next_frame)
    
    def psf_gaussian(self, size, sigma):
        """ Create a Gaussian PSF (point spread function). """
        k = cv2.getGaussianKernel(size, sigma)
        psf = np.outer(k, k)
        return psf

    def deconvolve_2d(self, image, psf, iterations=10):
        """ Deconvolve the image using the Wiener deconvolution approach. """
        image_ft = np.fft.fft2(image)
        psf_ft = np.fft.fft2(psf, s=image.shape)
        psf_ft_conj = np.conj(psf_ft)
        psf_ft_abs2 = np.abs(psf_ft) ** 2
        
        wiener_filter = psf_ft_conj / (psf_ft_abs2 + 0.01)
        deconvolved_ft = image_ft * wiener_filter
        deconvolved = np.fft.ifft2(deconvolved_ft)
        deconvolved = np.abs(deconvolved)
        
        return deconvolved

    def edge_aware_sharpen(self, image, psf_size=5, sigma=1.0):
        """ Perform edge-aware sharpening using deconvolution. """
        # Create a Gaussian PSF
        psf = self.psf_gaussian(psf_size, sigma)
        
        # Perform deconvolution
        sharpened_image = self.deconvolve_2d(image, psf)
        
        return np.clip(sharpened_image, 0, 255).astype(np.uint8)

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
            frame1 = cv2.imread('raw_data/raw_frames' + f'/gray_frame_{currentframe - 1}.jpg', cv2.IMREAD_GRAYSCALE)
            frame2 = cv2.imread('raw_data/raw_frames' + f'/gray_frame_{currentframe}.jpg', cv2.IMREAD_GRAYSCALE)

            frame_diff = self.compute_frames_difference(frame1, frame2)

            # Normalize the frame difference to span the full range of grayscale
            event_frame = cv2.normalize(frame_diff, None, 0, 255, cv2.NORM_MINMAX)
            event_frame = self.edge_aware_sharpen(event_frame)
            self.event_frames.append(event_frame)
            name = 'raw_data/event_frames/event_frame_' + str(currentframe - 1) + '.jpg'
            try:
                cv2.imwrite(name, event_frame)
            except:
                print("Error while writing event frame to file")
        self.event_frames = np.array(self.event_frames)
        print(f'{self.regular_frames_count} event frames created in \"raw_data/event_frames\" directory \n')
        self.event_frames_count = self.regular_frames_count
