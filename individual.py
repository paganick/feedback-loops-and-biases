from functions import get_truncated_normal

class individual():
    def __init__(self, theta, A, feature_noise_parameters):
        self.theta = theta
        self.A = A
        self.set_initial_x(feature_noise_parameters[A])
        
    def set_initial_x(self, feature_noise_parameters):
        self.x = get_truncated_normal(
            self.theta + feature_noise_parameters[0], feature_noise_parameters[1], 0, 1).rvs()
        
    