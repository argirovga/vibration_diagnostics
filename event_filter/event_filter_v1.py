import numpy as np
from scipy import signal
from deap import base, creator, tools, algorithms

from event_frames.event_frame_generator import EventFrameManager


EF_manager = EventFrameManager("raw_data/high_speed_cam_videos/MotorAmplif_motion_evm_2022-12-23-10-50-19.mp4")

# Extracting images from video
EF_manager.extract_frames()

# Applying grayscale filter to every frame
EF_manager.convert_to_grayscale()

# Creating event frames
EF_manager.create_event_frames()

# Ensure that the DEAP creator has not been previously used in this session
creator_name = "FitnessMax"
if creator_name in dir(creator):
    del creator.FitnessMax
    del creator.Individual

# Define the EventFilter class with IIR filter coefficients
class EventFilter:
    def __init__(self, b, a):
        self.b = b  # Numerator coefficients
        self.a = a  # Denominator coefficients

    def filter(self, data):
        # Apply IIR filter to data
        return signal.lfilter(self.b, self.a, data)

# Define the fitness function
def evaluate(individual, data):
    # Create filter instance from individual
    event_filter = EventFilter(individual[0], individual[1])
    # Apply filter to data and calculate fitness
    filtered_data = event_filter.filter(data)
    # Fitness could be the inverse of the sum of squared differences
    fitness = -np.sum((filtered_data - data) ** 2)
    return fitness,

# Set up the genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.rand)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=10)  # Modify 'n' as needed for the filter size
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate, data=np.array([]))  # Placeholder for the actual event frames data
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Initialize the genetic algorithm
population = toolbox.population(n=50)

# Define Hall of Fame and Statistics
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Run the genetic algorithm
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=True)

# Extract the best individual
best_filter = hof[0]
best_event_filter = EventFilter(best_filter[0], best_filter[1])

# Placeholder: Apply the best filter to your event frames data
# Replace 'your_event_frames_data' with the actual data
filtered_event_frames = [best_event_filter.filter(frame) for frame in EF_manager.event_frames]
