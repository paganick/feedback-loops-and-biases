#import random
from functions import get_truncated_normal


class feature_block():
    def __init__(self, feature_noise_parameters):
        self.feature_noise_parameters = feature_noise_parameters
        
    def r(self,individual):
        return individual.x
        #return get_truncated_normal(
        #    individual.theta + self.feature_noise_parameters[individual.A][0], self.feature_noise_parameters[individual.A][1], 0, 1).rvs()
    