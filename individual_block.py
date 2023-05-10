from individual import individual

class individual_block():
    def __init__(self):
        return    
        
    # Implementation of the g function:    
    def g(self,population, index):
        return population.get_individual(index)
        