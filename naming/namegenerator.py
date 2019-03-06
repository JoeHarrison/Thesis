import pandas as pd;
from collections import defaultdict
from random import choice
import numpy as np
from collections import Counter

class NameGenerator():
    def __init__(self, csv, markov_order, max_length):
        self.csv = csv
        self.df = pd.read_csv(csv)

        self.markov_order = markov_order
        
        self.max_length = max_length

        self.lowercase_names = self.lowercase_names()

        self.name_lengths = defaultdict(int)
        self.states = defaultdict(list)
        self.initialise_dicts()

        name_length_counter = Counter(self.name_lengths)
        total_names = sum(name_length_counter.values())
        self.name_probabilities = np.array(list(name_length_counter.values())) / total_names
        self.name_lengths = np.array(list(set(name_length_counter.elements()))) - self.markov_order

    def lowercase_names(self):
        return '<'*self.markov_order + self.df['name'].str.lower()

    def initialise_dicts(self):
        for name in self.lowercase_names:
            self.name_lengths[len(name)] += 1
            for x in range(self.markov_order,len(name)):
                self.states[tuple(name[x-self.markov_order:x])].append(name[x])

    def generate_name(self):
        n = 0
        while True:
            terms = list('<'*self.markov_order)
            for x in range(np.clip(np.random.choice(self.name_lengths,1,p=self.name_probabilities)[0],0,self.max_length)):
                if(self.states[tuple(terms[-self.markov_order:])]):
                    next_letter = choice(self.states[tuple(terms[-self.markov_order:])])
                    terms.append(next_letter)
                else:
                    break

            letters = terms[self.markov_order:]
            letters[0] = letters[0].capitalize()
            name = ''.join(letters)

            yield name
            n += 1
