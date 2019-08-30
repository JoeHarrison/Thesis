class Species(object):
    def __init__(self, new_specie_name, initial_member):
        self.name = next(new_specie_name)
        self.members = [initial_member]
        self.representative = initial_member
        self.offspring = 0
        self.age = 0
        self.average_fitness = 0.
        self.max_fitness = 0.
        self.max_fitness_previous = 0.0
        self.stagnation = 0
        self.has_best = False

    def reset_stagnation(self):
        self.stagnation = 0
        self.max_fitness_previous = 0.0
