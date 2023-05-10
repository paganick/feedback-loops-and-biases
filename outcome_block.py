#import random
from functions import get_truncated_normal, x2y

class outcome_block():
    def __init__(self,outcome_noise_parameters):
        self.outcome_noise_parameters = outcome_noise_parameters
        
    def t(self, individual):
        return x2y(individual.theta, self.outcome_noise_parameters[individual.A][0], self.outcome_noise_parameters[individual.A][1])
    
    def update(self):
        return