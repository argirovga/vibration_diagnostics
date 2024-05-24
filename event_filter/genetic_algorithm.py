import pygad
import numpy as np
from event_frames.event_frame_generator import EventFrameManager
from .event_filter_generator import EventFilter

import time

class GeneticAlgorithmCreator:
    def __init__(self, ef_manager: EventFrameManager, filter_order: int, num_generations: int, num_parents_mating: int, sol_per_pop: int,
                 init_range_low: float, init_range_high: float, mutation_percent_genes: float) -> None:
        self.ef_manager = ef_manager
        self.filter_order = filter_order
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        self.sol_per_pop = sol_per_pop
        self.init_range_low = init_range_low
        self.init_range_high = init_range_high
        self.mutation_percent_genes = mutation_percent_genes

    def fitness_function(self, ga_instance, solution, solution_idx):
        ef = EventFilter(self.ef_manager, self.filter_order, solution[:self.filter_order+1], solution[self.filter_order+1:])
        ef.generate_filtered_ef_frames()
        
        peak_intensities_sum_fitness = np.sum([np.max(frame) for frame in ef.filtered_event_frames])
        
        return peak_intensities_sum_fitness


    def create_run_instance(self, output_dir: str = None):
        num_genes = 2 * self.filter_order + 1
        ga_instance = pygad.GA(num_generations=self.num_generations,
                               num_parents_mating=self.num_parents_mating,
                               fitness_func=self.fitness_function,
                               sol_per_pop=self.sol_per_pop,
                               num_genes=num_genes,
                               init_range_low=self.init_range_low,
                               init_range_high=self.init_range_high,
                               mutation_percent_genes=self.mutation_percent_genes)
        start_time = time.time()
        ga_instance.run()
        convergence_time = round(time.time() - start_time, 2)
        solution, solution_fitness, _ = ga_instance.best_solution()
        print("Optimal Coefficients: ", solution)
        print("Fitness Value: ", solution_fitness)
        print("Convergence time in sec.: ", convergence_time)

        
        ef = EventFilter(self.ef_manager, self.filter_order, solution[:self.filter_order+1], solution[self.filter_order+1:])
        ef.generate_filtered_ef_frames(save_frames=True, output_dir=output_dir)

        return solution
