import torch.multiprocessing
from collections import defaultdict
from NEAT.species import Species
import random
import time
import numpy as np
from tqdm import tqdm

def evaluate_individual(item):
    (individual, evaluator, generation) = item
    if callable(evaluator):
        individual.stats = evaluator(individual, generation)
    elif hasattr(evaluator, 'evaluate'):
        individual.stats = evaluator.evaluate(individual, generation)
    else:
        raise Exception("Evaluator must be a callable or object" \
                    "with a callable attribute 'evaluate'.")
    return individual

class Population(object):
    def __init__(self, new_specie_name,
                 genome_factory,
                population_size = 100,
                elitism = True,
                stop_when_solved = False,
                tournament_selection_k = 3,
                verbose = True,
                max_cores = 1,
                compatibility_threshold = 3.0,
                compatibility_threshold_delta = 0.4,
                target_species = 12,
                minimum_elitism_size = 5,
                young_age = 10,
                young_multiplier = 1.2,
                old_age = 30,
                old_multiplier = 0.2,
                stagnation_age = 15,
                reset_innovations = False,
                survival = 0.2):

        self.new_specie_name = new_specie_name
        self.genome_factory = genome_factory
        self.population_size = population_size
        self.elitism = elitism
        self.stop_when_solved = stop_when_solved
        self.tournament_selection_k = tournament_selection_k
        self.verbose = verbose
        self.max_cores = max_cores
        
        cpus = torch.multiprocessing.cpu_count()
        use_cores = min(self.max_cores, cpus-1)
        if use_cores > 1:
            torch.multiprocessing.set_start_method('spawn')
            self.pool = torch.multiprocessing.Pool(processes=use_cores)
        else:
            self.pool = None
        
        self.compatibility_threshold = compatibility_threshold
        self.compatibility_threshold_delta = compatibility_threshold_delta
        
        self.target_species = target_species
        self.minimum_elitism_size = minimum_elitism_size
        
        self.young_age = young_age
        self.young_multiplier = young_multiplier
        self.old_age = old_age
        self.old_multiplier = old_multiplier
        
        self.stagnation_age = stagnation_age
        
        self.reset_innovations = reset_innovations
        self.survival = survival
        
    def _evaluate_all(self, population, evaluator):
        to_eval = [(individual, evaluator, self.generation) for individual in population]
        if self.pool is not None:
            population = list(self.pool.map(evaluate_individual, to_eval))
        else:
            population = list(map(evaluate_individual, to_eval))
        
        return population
        
    def _reset(self):
        self.champions = []
        self.generation = 0
        self.solved_at = None
        self.stats = defaultdict(list)
        self.species = []
        self.global_innovation_number = 0
        self.innovations = {}
        self.current_compatibility_threshold = self.compatibility_threshold

    def _reset_stagnation(self):
        print('resetting species')
        for specie in self.species:
            specie.reset_stagnation()
        
    def _find_best(self, population, solution = None):
        self.champions.append(max(population, key=lambda individual: individual.stats['fitness']))
        self.champions[-1].rl_training = True
        
        if solution is not None:
            if isinstance(solution, (int, float)):
                solved = (self.champions[-1].stats['fitness'] >= solution)
            elif callable(solution):
                solved = solution(self.champions[-1])
            elif hasattr(solution, 'solve'):
                solved = solution.solve(self.champions[-1])
                
            if solved and self.solved_at is None:
                self.solved_at = self.generation + 1
            
    @property
    def population(self):
        for specie in self.species:
            for member in specie.members:
                yield member
    
    def _evolve(self, evaluator, solution=None):
        population = list(self.population)
        
        while len(population) < self.population_size:
            individual = self.genome_factory()
            population.append(individual)        
            
        population = self._evaluate_all(population, evaluator)
        
        # Speciation
        for specie in self.species:
            # Choose random specie representative for distance comparison
            specie.representative = random.choice(specie.members)
            specie.name = specie.representative.specie
            specie.members = []
            specie.age += 1
            
        # Add each individual to a species
        reset_specie_flag = False
        for individual in population:
            if individual.stats['reset_species']:
                reset_specie_flag = True
            found = False
            for specie in self.species:
                if individual.distance(specie.representative) <= self.current_compatibility_threshold:
                    specie.members.append(individual)
                    individual.change_specie(specie.name)
                    found = True
                    break
            if not found:
                s = Species(self.new_specie_name, individual)
                individual.change_specie(s.name)
                self.species.append(s)

        if reset_specie_flag:
            self._reset_stagnation()
        
        # Remove empty species
        self.species = list(filter(lambda s: len(s.members) > 0, self.species))
        
        # Adjust compatibility threshold
        if len(self.species) < self.target_species:
            self.current_compatibility_threshold -= self.compatibility_threshold_delta
        elif len(self.species) > self.target_species:
            self.current_compatibility_threshold += self.compatibility_threshold_delta
        
        # Find champion and check for solution
        self._find_best(population, solution)
        
        # Recombination
        for specie in self.species:
            if specie.max_fitness > specie.max_fitness_previous:
                specie.max_fitness_previous = specie.max_fitness
            specie.average_fitness = np.mean([individual.stats['fitness'] for individual in specie.members])
            specie.max_fitness = np.max([individual.stats['fitness'] for individual in specie.members])
            if specie.max_fitness <= specie.max_fitness_previous:
                specie.stagnation += 1
            else:
                specie.stagnation = 0
            specie.has_best = self.champions[-1] in specie.members

        # Keep species that have the best or within stagnation age range
        self.species = list(filter(lambda s: s.stagnation < self.stagnation_age or s.has_best, self.species))
        
        # Adjust fitness based on age
        for specie in self.species:
            if specie.age < self.young_age:
                specie.average_fitness *= self.young_multiplier
            if specie.age > self.old_age:
                specie.average_fitness *= self.old_multiplier
                
        # Compute offspring size
        total_fitness = sum(specie.average_fitness for specie in self.species)
        for specie in self.species:
            if total_fitness == 0.0:
                specie.offspring = int(round(self.population_size/len(self.species)))
            else:
                specie.offspring = int(round(self.population_size * specie.average_fitness / total_fitness))
            
        
        
        # Remove species without offspring
        self.species = list(filter(lambda s: s.offspring > 0, self.species))

        for specie in self.species:
            specie.members.sort(key=lambda individual: individual.stats['fitness'], reverse = True)
            keep = max(1, int(round(len(specie.members)*self.survival)))
            pool = specie.members[:keep]
            
            if self.elitism and len(specie.members) > self.minimum_elitism_size:
                specie.members = specie.members[:1]
            else:
                specie.members = []
                
            while len(specie.members) < specie.offspring:
                k = min(len(pool), self.tournament_selection_k)
                p1 = max(random.sample(pool, k), key=lambda individual: individual.stats['fitness'])
                p2 = max(random.sample(pool, k), key=lambda individual: individual.stats['fitness'])
                
                child = p1.recombinate(p2)
                child.mutate(innovations=self.innovations, global_innovation_number = self.global_innovation_number)
                specie.members.append(child)
                
        if self.innovations:
            self.global_innovation_number = max(self.innovations.values())
            
        self._gather_stats(population)
            
    def epoch(self, evaluator, generations, solution=None, reset=True, callback= None):
        if reset:
            self._reset()
            
        for i in range(generations):
            self.time = time.time()
            self._evolve(evaluator, solution)
            self.generation += 1
            
            if self.verbose:
                self._status_report()
                
            if callback is not None:
                callback(self)
            
            if self.solved_at is not None and self.stop_when_solved:
                break
        
        return {'stats': self.stats, 'champions': self.champions}
    
    def _gather_stats(self, population):
        for key in population[0].stats:
            self.stats[key+'_avg'].append(np.mean([individual.stats[key] for individual in population]))
            self.stats[key+'_max'].append(np.max([individual.stats[key] for individual in population]))
            self.stats[key+'_min'].append(np.min([individual.stats[key] for individual in population]))
        self.stats['solved'].append(self.solved_at is not None)
    
    def _status_report(self):
        if self.verbose:
            print("\n****** Running Generation %d ******" % self.generation)
            print("****** Difficulty %d ******" % self.champions[-1].stats['info'])
            fitness_list = np.array([i.stats['fitness'] for i in self.population])
            number_neurons = len(self.champions[-1].neuron_genes)
            number_enabled_connections = np.sum([1 for conn in self.champions[-1].connection_genes.values() if conn[4]])
            print("Population's average fitness: %.5f stdev: %.5f" % (np.average(fitness_list), np.std(fitness_list)))
            print("Best individual: %s %s" % (self.champions[-1].name, self.champions[-1].specie))
            print("Best fitness: %.5f - #neurons: %i - #enabled connections: %i" % (self.champions[-1].stats['fitness'],number_neurons,number_enabled_connections))
            print("Population of %i members in %i species:" % (len(list(self.population)), len(self.species)))
            print("Species         age    size    fitness    stag")
            print("============    ===    ====    =======    ====")
            for specie in self.species:
                print("{: >12}    {: >3}    {: >4}    {:.5f}    {: >4}".format(specie.name, specie.age, len(specie.members), specie.max_fitness, specie.stagnation))
            print("Generation time: %.5f seconds" % (time.time()-self.time))
            print("Solved in generation: %s" % (self.solved_at))
