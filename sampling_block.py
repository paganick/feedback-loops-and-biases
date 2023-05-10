import population
import numpy as np

class sampling_block():
    def __init__(self):
        return
    
    def s(self, population):
        i = np.random.randint(0, population.n-1)
        return i 
        