from event_frames.event_frame_generator import EventFrameManager
from event_filter.event_filter_generator import EventFilter
from event_filter.genetic_algorithm import GeneticAlgorithmCreator

import numpy as np

# Define the file path for the video
file_path = "raw_data/high_speed_cam_videos/MotorAmplif_motion_evm_2022-12-23-10-50-19.mp4"

# Initialize the EventFrameManager to extract frames from the video
EF_manager = EventFrameManager(file_path)

# Create an instance of EventFilter to generate filtered event frames
filter_order = 2  # Example filter order, adjust as needed
b_coefficients = np.array([0.05, 0.05, 0.05])  # Example feedforward coefficients, adjust as needed
a_coefficients = np.array([0.05, 0.05])  # Example feedback coefficients, adjust as needed
EF = EventFilter(EF_manager, filter_order, b_coefficients, a_coefficients)
EF.generate_filtered_ef_frames()

# Create an instance of GeneticAlgorithmCreator to optimize the EF coefficients
num_generations = 50
num_parents_mating = 4
sol_per_pop = 8
init_range_low = -2
init_range_high = 5
mutation_percent_genes = 50
GA_creator = GeneticAlgorithmCreator(EF_manager, filter_order, num_generations, num_parents_mating, sol_per_pop,
                                     init_range_low, init_range_high, mutation_percent_genes)
optimized_coefficients = GA_creator.create_run_instance('preped_data/filtered_event_frames')

 